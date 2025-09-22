# this code serves for univariate time series only

import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import faiss
import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.models import vgg16, VGG16_Weights
import argparse

def init_model(num_first_layer):
		weights = VGG16_Weights.DEFAULT  # or .IMAGENET1K_V1
		vgg16_model = vgg16(weights=weights).eval()   # can change by ResNet, DenseNet, ViT,etc.
		first_layers = nn.Sequential(*list(vgg16_model.features.children())[:num_first_layer])
		return first_layers

def embedding(input_tensor, first_layers=None):
	
	output_tensor = first_layers(input_tensor)
	# Max pooling
	#output_tensor = nn.MaxPool2d(kernel_size=2, stride=2)(output_tensor)
	# Flatten
	print(output_tensor.shape)
	output_tensor = output_tensor.view(output_tensor.size(0), -1)
	print(output_tensor.shape)
	return output_tensor

def main():
	parser = argparse.ArgumentParser(description="Retrieval process with GAF + FAISS")
	parser.add_argument("--num_first_layers", type=int, nargs="+", required=True, help="number of first layers used in prertrained vision encoder") # 3 4 5
	parser.add_argument("--his_len_list", type=int, nargs="+", required=True, help=" his_len list") # [20, 40, 60, 80, 100]
	parser.add_argument("--symbol_list", type=str, nargs='+', help="symbol") #AAPL CSCO, XOM,...
	parser.add_argument("--device", type=str, default="cuda", help="device")
	parser.add_argument("--database_list", type=str, nargs='+', default=['only'], help="database list") # in this paper, database is default 'only'
	parser.add_argument("--step_size_list", type=int, nargs="+", default=[1], help="step size list") # read paper to have full understand about this hyperparameter
	args = parser.parse_args() 

	

	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	for step_size in args.step_size_list:
		for num_first_layer in args.num_first_layers:
			first_layers = init_model(num_first_layer=num_first_layer)
			for his_len in args.his_len_list:
				for symbol in args.symbol_list:
					for database in args.database_list:
						total_len = 2 * his_len  # Total length of the sequence ( his_len == prediction length)
						input_file = f'./raw_data/{symbol}.csv'
						

						df = pd.read_csv(input_file, usecols=['Date', 'Symbol', 'Close'])
						df['Date'] = pd.to_datetime(df['Date'])
						df = df.sort_values(['Symbol', 'Date']) 

						df['Close'] = df['Close'].astype(float)
						all_sequences = []

						
						for symbol in tqdm(df['Symbol'].unique(), desc="run symbol"):
							df_symbol = df[df['Symbol'] == symbol].copy()
							close_prices = df_symbol['Close'].values
							if len(close_prices) < total_len:
								continue 
							normalized = close_prices.copy()

							# GAF
							gasf = GramianAngularField(image_size=his_len, method='summation')
							gadf = GramianAngularField(image_size=his_len, method='difference')
							tensor_images = []
							window_list = []
								
							for i in range(len(normalized) - total_len + 1):
								window = normalized[i:i + his_len].reshape(1, -1)
								window_list.append(normalized[i:i + 2 * his_len].reshape(1, -1))
								# Generate GAF images
								gasf_img = gasf.transform(window)[0]
								gadf_img = gadf.transform(window)[0]
								zeros_img = np.zeros_like(gadf_img)
			
								stacked = np.stack([gasf_img, gadf_img, zeros_img], axis=-1)
								tensor_img = torch.tensor(stacked.transpose(2, 0, 1), dtype=torch.float32)
								tensor_images.append(tensor_img)
							tensor_images = torch.stack(tensor_images, dim=0)
							batch_emb = embedding(tensor_images, first_layers=first_layers)
							top_k = 10  # this value can be adjusted to a larger one; it also covers cases where k < top_k (e.g., taking the top 5 from top_k = 10)
							margin = his_len
							N = len(batch_emb)
							print(N)
							# split train and test
							split = int(0.7 * N)
							all_topk_windows = []
							norm_batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
							d = norm_batch_emb.shape[1]
							for i in range(N):
								valid_indices = [j for j in range(N) if abs(j - i) > margin and j <= split and (j - i) % step_size == 0]
								if len(valid_indices) < top_k:
									continue
								query_emb = norm_batch_emb[i].unsqueeze(0).cpu().detach().numpy().astype('float32')  # (1, d)
								ref_emb = norm_batch_emb[valid_indices].cpu().detach().numpy().astype('float32')     # (M, d)
								
								index = faiss.IndexFlatIP(d)  # inner product
								index.add(ref_emb)

								# Search top_k
								scores, idxs = index.search(query_emb, top_k)  # (1, top_k)

								top_scores = torch.from_numpy(scores[0])      # cosine similarity
								top_indices = idxs[0]                         # index in valid_indices
								#print(top_scores)                      
								selected_windows = [window_list[valid_indices[j]] for j in top_indices]
								
								for win in selected_windows:
									all_topk_windows.append(win.flatten())
							concat_tensor = torch.tensor(np.stack(all_topk_windows), dtype=torch.float32).flatten()
							print(concat_tensor.shape)
							output_file = f'./Database/{symbol}/VGG16/{num_first_layer}/{database}_gasf_gadf/{step_size}/{his_len}.pt'
							os.makedirs(os.path.dirname(output_file), exist_ok=True)
							torch.save(concat_tensor, output_file)

if __name__ == "__main__":
	main()

				
			  