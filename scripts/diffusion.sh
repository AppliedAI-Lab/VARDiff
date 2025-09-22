#!/bin/bash -l
cd VARDiff
python clear_cuda.py

# Experiment configurations
seeds=(2025)
symbols=('CSCO' 'JNJ')
delay=(3 8 30) #30
img_resolution=(32)
unet_channels=(32 64)
ch_mult=("1 2") #("1 2" "1 2 4")
seq_len=(20)
diffusion_steps=(50)
attn_resolution=("8 4" "16 8") #("16 8 4 2" "16 8 4") "16 8"
channel_mult_emb=(2)
num_blocks=(2) 
top_k=(3 4)
step_sizes=(1)
dropout=(0.15 0.2)
convert_method=("gasf_gadf")
num_first_layer=(4) # 3 4 5
pretrained_model=('VGG16')
batch_size=(4 8)
N=2  # number of parallel jobs

sem() {
    while (( $(jobs -rp | wc -l) >= N )); do
        sleep 1
    done
    "$@" &
}

# Loop through combinations
for seed in "${seeds[@]}"; do
    for delay_val in "${delay[@]}"; do
        for dropout_val in "${dropout[@]}"; do
            for seq_len_val in "${seq_len[@]}"; do
                for diffusion_steps_val in "${diffusion_steps[@]}"; do
                    for attn_res_val in "${attn_resolution[@]}"; do
                        for ch_mult_val in "${ch_mult[@]}"; do
                            for unet_channels_val in "${unet_channels[@]}"; do
                                for img_resolution_val in "${img_resolution[@]}"; do
                                    for convert_method_val in "${convert_method[@]}"; do
                                        for step_sizes_val in "${step_sizes[@]}"; do
                                            for top_k_val in "${top_k[@]}"; do
                                                for channel_mult_emb_val in "${channel_mult_emb[@]}"; do
                                                    for num_blocks_val in "${num_blocks[@]}"; do
                                                        for num_first_layer_val in "${num_first_layer[@]}"; do
                                                            for batch_size_val in "${batch_size[@]}"; do
                                                                for symbol in "${symbols[@]}"; do
                                                                    symbol_lower=$(echo "$symbol" | tr '[:upper:]' '[:lower:]')
                                                                    for pretrained_model_val in "${pretrained_model[@]}"; do

                                                                
                                                                            echo "Launching: $log_path"
                                                                            echo "Symbol: $symbol | Seed: $seed | Delay: $delay_val | Seq: $seq_len_val | Steps: $diffusion_steps_val"
                                                                            echo "Attn Res: $attn_res_val | Ch Mult: $ch_mult_val | UNet Ch: $unet_channels_val"
                                                                            echo "----------------------------------------------------"

                                                                            sem python -u run_conditional.py \
                                                                                --symbols "$symbol" \
                                                                                --config ./configs/extrapolation/${symbol_lower}.yaml \
                                                                                --seq_len "$seq_len_val" \
                                                                                --batch_size "$batch_size_val" \
                                                                                --seed "$seed" \
                                                                                --epochs 1000 \
                                                                                --convert_method "$convert_method_val" \
                                                                                --patience 5 \
                                                                                --top_k "$top_k_val" \
                                                                                --step_sizes "$step_sizes_val" \
                                                                                --delay "$delay_val" \
                                                                                --img_resolution "$img_resolution_val" \
                                                                                --unet_channels "$unet_channels_val" \
                                                                                --embedding "$img_resolution_val" \
                                                                                --diffusion_steps "$diffusion_steps_val" \
                                                                                --attn_resolution $attn_res_val \
                                                                                --ch_mult $ch_mult_val \
                                                                                --channel_mult_emb "$channel_mult_emb_val" \
                                                                                --num_blocks "$num_blocks_val" \
                                                                                --dropout "$dropout_val" \
                                                                                --num_first_layer "$num_first_layer_val" \
                                                                                --pretrained_model "$pretrained_model_val"
                                                                            
                                                                        
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
wait
echo "All jobs started in background. Logs in logs/"

