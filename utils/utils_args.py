from omegaconf import OmegaConf
import argparse

def parse_args_cond():
    """
    Parse arguments for conditional models
    Returns: conditional generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'conditional'],
                        help='tags for neptune logger', nargs='+')

    # --- diffusion process ---
    #parser.add_argument('--reference', type=str, help='the link of the reference time series')
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='deterministic sampling')

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/interpolation/TS2I/physionet.yaml',
                        help='config file')

    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    
    parser.add_argument('--symbols', type=str,
                        help='list of symbol (Ex: GOOG AMZN AAPL)')
    parser.add_argument('--run_type', type=str, default='only',
                        choices=['only'],
                        help='only type for this paper')
    # --- data ---
    parser.add_argument('--dataset',
                        choices=['kdd_cup', 'traffic_hourly', 'solar_weekly', 'temperature_rain',
                                 'nn5_daily', 'fred_md', 'sine', 'energy', 'mujoco', 'stocks', 'goog', 'aapl', 'amzn'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')
    
    # Database
    parser.add_argument('--top_k', type=int, default=10,
                    help='the number of reference retrieved')
    parser.add_argument('--step_sizes', type=int,
                        help='read paper to have full understand about this hyperparameter (T_s)')
    parser.add_argument('--convert_method', type=str, default='gasf_gadf',
                        choices=['gasf_gadf', 'gasf_gadf_difference', 'gasf_gadf_linear_trend'],
                        help='T_1 convert method to transform time series to images')
    parser.add_argument('--pretrained_model', type=str, default='VGG16',
                        help='pretrained vision encoder to extract features from images')
    parser.add_argument('--num_first_layer', type=int, default= 4,
                        help = 'number of first layers used in prertrained vision encoder')
    
    
    # --- image transformations ---
    parser.add_argument('--use_stft', type=bool,
                        help='use stft transform - if absent, use delay embedding')  # can be base
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')

    # --- model---
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--channel_mult_emb', type=int, help='channel mult emb', nargs='+')
    parser.add_argument('--num_blocks', type=int, help='number of blocks in the unet')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for the model')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, default=5,
                        help='number of iterations between logging')

    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    return parsed_args

