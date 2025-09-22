# this code serves for multivariate time series. In this file, it builds for Etth dataset, we totally apply for any multivariate time series dataset

import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LinearRegression
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.nn.functional import cosine_similarity
import faiss
import torchvision.models as models
import torch.nn as nn
import random
import torch
from torchvision.models import vgg16, VGG16_Weights
import time
import argparse

# his_len_list = [80, 100] #[10,20,40, 60]
# symbol_list = ['ETTh1'] #, 'AAPL',  'AMZN', 'GEV', 'JPM', 'LIN', 'LLY', 'NEE', 'PG', 'PLD', 'XOM'
# database_list = ['only'] #['only', 'industry', 'all']
# num_first_layers = [3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def init_model(num_first_layer):
		weights = VGG16_Weights.DEFAULT  # hoặc .IMAGENET1K_V1
		vgg16_model = vgg16(weights=weights).eval()
			
		first_layers = nn.Sequential(*list(vgg16_model.features.children())[:num_first_layer]).to(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		return first_layers

def embedding(input_tensor, first_layers=None):
	
	output_tensor = first_layers(input_tensor)
	# Max pooling
	#output_tensor = nn.MaxPool2d(kernel_size=2, stride=2)(output_tensor)
	# Flatten
	output_tensor = output_tensor.view(output_tensor.size(0), -1)

	
	return output_tensor

def embedding_in_batches(tensor_images, first_layers, batch_size=32):
	all_embs = []
	with torch.no_grad():
		for start in range(0, len(tensor_images), batch_size):
			batch = tensor_images[start:start+batch_size].to(device)
			with torch.no_grad(): 
				emb = embedding(batch, first_layers=first_layers)
				all_embs.append(emb.cpu())
				del batch, emb
				torch.cuda.empty_cache()
	return torch.cat(all_embs, dim=0)

def main():
	parser = argparse.ArgumentParser(description="Retrieval process with GAF + FAISS")
	parser.add_argument("--num_first_layers", type=int, nargs="+", required=True, help="number of first layers used in prertrained vision encoder")
	parser.add_argument("--his_len_list", type=int, nargs="+", required=True, help=" his_len list")
	parser.add_argument("--symbol", type=str, required=True, help="symbol")
	parser.add_argument("--device", type=str, default="cuda", help="device")
	parser.add_argument("--step_size_list", type=int, nargs="+", default=[1], help="step size list") # read paper to have full understand about this hyperparameter
	args = parser.parse_args()
	

	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	for num_first_layer in args.num_first_layers:
		first_layers = init_model(num_first_layer=num_first_layer)
		for his_len in args.his_len_list:

					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()
					total_len = 2 * his_len  # Total length of the sequence (history + prediction length)
					input_file = f'./data/short_range/ETT-small/{args.symbol}.csv'
					
					df = pd.read_csv(input_file)
					cols = [c for c in df.columns if c != 'date']
					concat_tensor = [] 
					for col in cols:
						df_c = pd.read_csv(input_file, usecols=['date', col])
						df_col = df_c[[col]].astype(float)
						all_sequences = []
						col_value = df_col.copy()
						
						if len(col_value) < total_len:
							continue 
						normalized = col_value
						train_data = normalized[0: 12 * 30 * 24]
						scaler = StandardScaler()
						scaler.fit(train_data.values)
						normalized = scaler.transform(normalized.values)

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
						
						batch_emb = embedding_in_batches(tensor_images, first_layers=first_layers)
						del tensor_images
						torch.cuda.empty_cache()
						
						top_k = 10
						margin = his_len
						N = len(batch_emb)
						print(N)
						# split train and test
						split = 12 * 30 * 24 - his_len
						
						all_topk_windows = []
						norm_batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
						d = norm_batch_emb.shape[1] 
						
						batch_emb_np = batch_emb.cpu().numpy().astype('float32')
						faiss.normalize_L2(batch_emb_np)
						#  index Faiss Inner Product
						index = faiss.IndexFlatIP(d)
						index.add(batch_emb_np)  

						search_k = top_k + 2 * margin + split
						scores_all, idx_all = index.search(batch_emb_np, search_k)
						selected_windows_all = []
						for i in range(N):
							mask = [
								j for j in idx_all[i]
								if abs(j - i) > margin and j < split
							]
							if len(mask) >= top_k:
								selected_windows_all.append([window_list[j] for j in mask[:top_k]])
						
						col_tensor = torch.tensor(np.stack(selected_windows_all), dtype=torch.float32).flatten()
						concat_tensor.append(col_tensor)

					concat_tensor = torch.stack(concat_tensor, dim=0)
					print(concat_tensor.shape)
					output_file = f'./Database_ETT/ETT/{args.symbol}/VGG16/{num_first_layer}/only_gasf_gadf/{his_len}.pt'
					os.makedirs(os.path.dirname(output_file), exist_ok=True)
					torch.save(concat_tensor, output_file)
					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()

if __name__ == "__main__":
	main()



				
				