import numpy as np
import torchaudio.transforms as transforms
import os
import pandas as pd
import sys
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler as Ori_MinMaxScaler
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset
from data.data_provider.data_factory import data_provider
# from data.long_range import parse_datasets 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def real_data_loading(args, data_name, seq_len):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """

    data_name_upper = data_name.upper()
    ori_data = np.loadtxt(f'./data/short_range/{data_name_upper}.csv', delimiter=",", skiprows=1)


    ori_data = torch.Tensor(ori_data)  # shape [N]

    train_ratio = 0.7
    train_size = int(len(ori_data) * train_ratio)
    train_data = ori_data[:train_size]
    test_data = ori_data[train_size:]

    scaler = Ori_MinMaxScaler()
    
    train_data = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
    test_data = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
    # Save mean and std 
    mean, std = scaler.data_min_, scaler.data_max_ - scaler.data_min_
    args.mean, args.std = torch.Tensor(mean), torch.Tensor(std)   

    train_set = []
    test_set = []
    # Cut data by sequence length
    for i in range(0, len(train_data) - seq_len + 1):
        _x = train_data[i:i + seq_len]
        train_set.append(_x)
    for i in range(0, len(test_data) - seq_len + 1):
        _x = test_data[i:i + seq_len]
        test_set.append(_x)

    del ori_data, train_data, test_data
    torch.cuda.empty_cache()

    return train_set, test_set


def normalize(data, mean=None, std=None):
    return (data - mean) / (std + 1e-7)
def denormalize(data, mean, std):
    return data * std + mean

def gen_dataloader(args):

    if args.dataset in ['goog', 'amzn', 'aapl', 'energy', 'jpm', 'nee','xom', 'pg', 'ge', 'jnj', 'csco', 'msft', 'jpm']:
        train_data, test_data = real_data_loading(args, args.dataset, args.seq_len)
        
        # reference = f"./Database/{args.symbols}/{args.pretrained_model}/{args.num_first_layer}/{args.run_type}_gasf_gadf/{args.step_sizes}/{args.seq_len // 2}.pt"
        reference = f"./Database/{args.symbols}/{args.pretrained_model}/{args.num_first_layer}/{args.run_type}_gasf_gadf/{args.seq_len // 2}.pt"

        ref = torch.load(reference)
        print(ref.shape)
        
        train_data = torch.Tensor(np.array(train_data))
        test_data = torch.Tensor(np.array(test_data))
        num_train = len(train_data)
        num_test = len(test_data)
        train_ref_data = []
        test_ref_data = []
        ref_all = ref.view(-1, 10 * args.seq_len) # if change top_k in retrieval, need to change here
        train_data = torch.tensor(np.array(train_data), dtype=torch.float32)
        # Train part
        train_ref = ref_all[:num_train, :args.top_k * args.seq_len]
        train_ref = (train_ref - args.mean) / args.std

        train_ref_data = torch.cat([train_data, train_ref], dim=1)
        test_data = torch.tensor(np.array(test_data), dtype=torch.float32)

        # Index offset cho test
        start = num_train + args.seq_len - 1
        end = start + num_test

        test_ref = ref_all[start:end, :args.top_k * args.seq_len]
        test_ref = (test_ref - args.mean) / args.std

        test_ref_data = torch.cat([test_data, test_ref], dim=1)

        del train_data, test_data, ref_all, train_ref, test_ref, ref
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # create TensorDataset and DataLoader
        train_set = Data.TensorDataset(train_ref_data)
        test_set = Data.TensorDataset(test_ref_data)
        
        train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=False)

        test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                                num_workers= args.num_workers, drop_last=False)

        

        return train_loader, test_loader

    elif args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        train_data, train_loader = data_provider(args, flag='train')
        test_data, test_loader = data_provider(args, flag='test')
        return train_loader, test_loader


    # for the short-term time series benchmark, the entire dataset for both training and testing
    return train_loader, train_loader





def stft_transform(data, args):
    data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
    n_fft = args.n_fft
    hop_length = args.hop_length
    spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=None)
    transformed_data = spec(data)
    real, min_real, max_real = MinMaxScaler(transformed_data.real.numpy(), True)
    real = (real - 0.5) * 2
    imag, min_imag, max_imag = MinMaxScaler(transformed_data.imag.numpy(), True)
    imag = (imag - 0.5) * 2
    # saving min and max values, we will need them for inverse transform
    args.min_real, args.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
    args.min_imag, args.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
    return torch.Tensor(real), torch.tensor(imag)


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


