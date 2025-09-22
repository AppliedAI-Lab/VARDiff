import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy, csv, wandb
#wandb.login(key = 'your-wandb-key')
wandb.login(key = 'c9bf0410a696f6094571b4bcf35e7f93c95fe86d')
import torch
import time
import numpy as np
import torch.multiprocessing
import logging, copy
import torch.nn.functional as F
from utils.tools import EarlyStopping
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
	log_config_and_tags, get_x_and_mask
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_cond
from utils.utils_vis import visual
import os
import pandas as pd
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
#torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
	# model name and directory
	name = create_model_name_and_dir(args)

	# log args
	logging.info(args)
	# set-up neptune logger. switch to your desired logger
	with CompositeLogger([NeptuneLogger()]) if args.neptune \
			else PrintLogger() as logger:

		# log config and tags
		log_config_and_tags(args, logger)
		
		# --- set-up data and device ---
		args.device = "cuda" if torch.cuda.is_available() else "cpu"
		symbols = args.symbols
		run_type = args.run_type
		top_k = args.top_k
		step_size = args.step_sizes
		convert_method = args.convert_method
		train_loader, test_loader = gen_dataloader(args) 
		print(f"Train loader length: {len(train_loader)}, Test loader length: {len(test_loader)}")
		#reference = args.reference if args.reference else None
		#ref = torch.load(reference)
		#print(f"Shape of ref: {ref.shape}")  # (batch_size, seq_len, top_k)


		logging.info(args.dataset + ' dataset is ready.')
		
		
				
					
		wandb.init(project=f"Unet_{args.epochs}_{args.top_k}_{step_size}", name="checking")
		early_stopping = EarlyStopping(patience=args.patience, verbose=True)
		path = os.path.join('checkpoints', args.symbols)
		if not os.path.exists(path):
			os.makedirs(path)

		# update args
		local_args = copy.deepcopy(args)
			# (batch_size, seq_len, top_k)
		# update local_args with specific parameters
		local_args.convert_method = convert_method
		local_args.top_k = top_k
		local_args.step_size = step_size
		main_model = ImagenTime(args=local_args, device=local_args.device).to(local_args.device)

		# optimizer
		trainable_params = filter(lambda p: p.requires_grad, main_model.parameters())
		optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

		state = dict(model=main_model, epoch=0)
		init_epoch = 0

		# restore checkpoint
		if args.resume:
			ema_model = main_model.model_ema if args.ema else None # load ema model if available
			init_epoch = restore_state(args, state, ema_model=ema_model)

		# print model parameters
		#print_model_params(logger, model)

		# --- train model ---
		logging.info(f"Continuing training loop from epoch {init_epoch}.")
		
		best_score_mae = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
		best_score_mse = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
		for epoch in range(init_epoch, args.epochs):
			print(f"Epoch {epoch + 1}/{args.epochs}")
			main_model.train()
			main_model.epoch = epoch
			train_losses = []
			eval_losses = []

			# --- train loop ---
			total_samples = 0 # the number of sample used
			for i, data in enumerate(train_loader, 0): #1
				# if i == 1:
				#     break
				x = data[0]  # shape: [batch_size, total_seq_len]
				batch = len(x)     ## because last batch can not enough samples (!= batch size)
				ref = x[:, local_args.seq_len:]  # shape: [batch_size, top_k * seq_len]

				x_input = x[:, :local_args.seq_len]  # shape: [batch_size, seq_len]
				
				# create mask & x_ts
				mask_ts, x_ts = get_x_and_mask(args, x_input)
				if mask_ts.dim() == 2:  #(B, seq_len,) univariate time series
						mask_ts = mask_ts.unsqueeze(-1)  # (B, seq_len, 1)
						x_ts = x_ts.unsqueeze(-1)  # (B, seq_len, 1)
						ref = ref.unsqueeze(-1)
						

				x_ref = ref.reshape(batch, local_args.top_k, local_args.seq_len, ref.shape[-1]).to(local_args.device)
				sample_img = main_model.ts_to_img(x_ref[:, 0])  # shape: (batch_size, C, H, W)

				B, C, H, W = sample_img.shape # B = batch size, C = features, H = height, W = width
				x_ref_ts_img = torch.zeros((batch, args.top_k, C, H, W), device=args.device)
				
				b, k = x_ref.shape[:2]

				x_ref_ts_img = main_model.ts_to_img(
					x_ref.reshape(b * k, *x_ref.shape[2:]) 
				)  # → (b*k, C, H, W)
				# transform to image
				x_ts_img = main_model.ts_to_img(x_ts)
		
				# pad mask with 1
				mask_ts_img = main_model.ts_to_img(mask_ts,pad_val=1)
				optimizer.zero_grad()  
				loss = main_model.loss_fn_impute(x_ts_img, mask_ts_img, ref = x_ref_ts_img, top_k=args.top_k)
				del x_ts_img, mask_ts_img, x_ref_ts_img
				torch.cuda.empty_cache()
				torch.cuda.ipc_collect()
				if len(loss) == 2:
					loss, to_log = loss
				
				loss.backward()
				torch.nn.utils.clip_grad_norm_(main_model.parameters(), 1.)
				optimizer.step()
				main_model.on_train_batch_end()
				train_losses.append(loss.item())
			
			avg_train_loss = sum(train_losses) / len(train_losses)
			wandb.log({"train/loss": avg_train_loss}, step=epoch)
				

			# --- evaluation loop ---
			# best_score_mae = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
			# best_score_mse = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
			if epoch % args.logging_iter == 0:
				mse = 0
				mae = 0
				main_model.eval()
				
				with torch.no_grad():
					with main_model.ema_scope():
						process = DiffusionProcess(args, main_model.net,
												(args.input_channels, args.img_resolution, args.img_resolution))
						
						j = len(train_loader)
						
						
						print("len(test_loader):", len(test_loader))
						
						for idx, data in enumerate(test_loader, 0): 
							batch = len(data[0])

							x = data[0]  # shape: [batch_size, total_seq_len]
							batch = len(x)     ## because last batch can not enough samples (!= batch size)
							x_input = x[:, :local_args.seq_len]  # shape: [batch_size, seq_len]
							ref = x[:, local_args.seq_len:]  # shape: [batch_size, top_k * seq_len]
							mask_ts, x_ts = get_x_and_mask(args, x_input)
							if mask_ts.dim() == 2:  #(B, seq_len,) univariate time series
								mask_ts = mask_ts.unsqueeze(-1)  # (B, seq_len, 1)
								x_ts = x_ts.unsqueeze(-1)  # (B, seq_len, 1) # After this, x_ts is multiplied with a mask to ensure the prediction part does not leak
								ref = ref.unsqueeze(-1)
							x_ref = ref.reshape(batch, local_args.top_k, local_args.seq_len, ref.shape[-1]).to(local_args.device)


							sample_img = main_model.ts_to_img(x_ref[:, 0])  # (find shape): (batch_size, C, H, W)
							B, C, H, W = sample_img.shape # B = batch size, C = features, H = height, W = width
							x_ref_ts_img = torch.zeros((batch, args.top_k, C, H, W), device=args.device)

							b, k = x_ref.shape[:2]

							x_ref_ts_img = main_model.ts_to_img(
								x_ref.reshape(b * k, *x_ref.shape[2:])
							)  # → (b*k, C, H, W)
							x_ts_img = main_model.ts_to_img(x_ts)

							mask_ts_img = main_model.ts_to_img(mask_ts, pad_val=1)
							# sample from the model
							# and impute, both interpolation and extrapolation are similar just the mask is different
						
							x_img_sampled = process.interpolate(x_ts_img, mask_ts_img, ref = x_ref_ts_img).to(x_ts_img.device)
							del x_ts_img, mask_ts_img, x_ref_ts_img
							torch.cuda.empty_cache()
							torch.cuda.ipc_collect()
							x_ts_sampled = main_model.img_to_ts(x_img_sampled)
							# task evaluation
							x_ts_pred = x_ts_sampled
							mse_mean = F.mse_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_pred[mask_ts == 0])
							
							mae_mean = F.l1_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_pred[mask_ts == 0])		
							mse += mse_mean.item()
							mae += mae_mean.item()

				scores = {'mse': mse / (idx + 1), 'mae': mae / (idx + 1)}
				eval_losses.append(scores['mse']) # use for wandb
				vali_loss = scores['mse'] # use for Early stopping
				print(f"Epoch {epoch}, MSE: {scores['mse']}, MAE: {scores['mae']}")
				early_stopping(vali_loss, main_model, path)
				if early_stopping.early_stop:
					print("Early stopping")
					break
				

				# --- save checkpoint ---
				curr_score_mse = scores['mse']
				curr_score_mae = scores['mae']
				if curr_score_mse < best_score_mse:
					best_score_mse = curr_score_mse
					best_score_mae = curr_score_mae
					print(f" New best at epoch {epoch}, top k {local_args.top_k}, step size {step_size}: MSE={best_score_mse:.4f}, MAE={best_score_mae:.4f}")
					ema_model = main_model.model_ema if args.ema else None
				#save_checkpoint(args.log_dir, state, epoch, ema_model)
		
		print(f"Symbol: {local_args.symbols}, Seq_len: {local_args.seq_len}, Top k: {local_args.top_k}, Step size: {local_args.step_size}, Best MSE: {best_score_mse}, Best MAE: {best_score_mae}")
		
		
		filename_csv = "logs/final.csv"
		if not os.path.exists(filename_csv):
			with open(filename_csv, mode='w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(['architecture','seed', 'diffusion steps', 'symbols', 'database', 'CLIP model', 'ts2img (retrieval)', 'ts2img( Unet flow)', 'unet channel',  'delay', 'img resolution', 'ch_mult', 'attn_resolution', 'channel_mult_emb', 'num_blocks', 'history len', 'pred len', 'top k', 'step size', 'batch size', 'epochs', 'Best_MSE', 'Best_MAE', 'Dropout', 'num_first_layer', 'pretrained_model'])

		data = {
			"attention architecture": 'Decoder-only',
			"seed": [local_args.seed],
			"diffusion steps": [local_args.diffusion_steps],
			"symbols": [local_args.symbols],
			"database": [local_args.run_type],
			'ts2img (retrieval)':[local_args.convert_method],
			'ts2img( Unet flow)': 'Delay Embedding',
			'unet channel': [local_args.unet_channels],
			'delay': [local_args.delay],
			'img resolution': [local_args.img_resolution], 
			'ch_mult': [local_args.ch_mult],
			'attn_resolution': [local_args.attn_resolution],
			'channel_mult_emb': [local_args.channel_mult_emb],
			'num_blocks': [local_args.num_blocks],                       
			'history len': [local_args.seq_len // 2],
			'pred len':[local_args.seq_len // 2],
			'top k': [local_args.top_k], 
			'step size': [local_args.step_size],
			'batch size': [args.batch_size], 
			'epochs': [local_args.epochs], 
			'Best_MSE': [best_score_mse],
			'Best_MAE': [best_score_mae],
			'Dropout': [local_args.dropout],
			'Num first layer': [local_args.num_first_layer],
			'pretrained model': [local_args.pretrained_model]
		}
		df = pd.DataFrame(data)
		if os.path.exists(filename_csv):
			df.to_csv(filename_csv, mode='a', index=False, header=False)
		else:
			df.to_csv(filename_csv, index=False)

	logging.info("Training is complete")


if __name__ == '__main__':
	args = parse_args_cond() 
	torch.random.manual_seed(args.seed)
	np.random.default_rng(args.seed)
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
	main(args)

