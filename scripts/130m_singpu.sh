#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=130m_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=./logs/130m_test.out

# module purge
# module load 2023
#source activate spam
# Your job starts in the directory where you call sbatch
# cd ../
# Activate your environment
# Run your code
echo "Running experiment on galore..."
START_TIME=`date`; echo ">>> START: $START_TIME"

# Check whether the GPU is available
python -uc "import torch; print('>>> GPU available?', torch.cuda.is_available())"

save_dir_base=./checkpoints
# Create a unique save directory by appending the date and time
current_datetime=$(date +"%Y%m%d_%H%M%S")
save_dir="${save_dir_base}/${current_datetime}"

export OMP_NUM_THREADS=8


num_training_steps=20000
opt="Adam"
for prj in 150
do
  for lr in 0.0008 #0.003 #0.0008 #0.0001 0.001 0.003
  do
      # default lr: 8e-4
      # default num_training_steps 20_000
      echo ">>> OPTIMIZER: $opt"
      echo ">>> LR: $lr"
      torchrun --standalone --nnodes 1 --nproc_per_node 1 torchrun_main.py \
          --project_name "130m_${opt}_${lr}_gradspikes" \
          --model_config configs/llama_130m.json \
          --lr $lr \
          --density 1.0 \
          --update_gap 500 \
          --batch_size 32  \
          --total_batch_size 512 \
          --num_training_steps $num_training_steps \
          --warmup_steps 1000 \
          --weight_decay 0 \
          --dtype bfloat16 \
          --eval_every 1000 \
          --threshold 5000 \
          --save_dir $save_dir \
          --optimizer $opt \
          --warmup_epoch $prj \
          --single_gpu
  done
done

#num_training_steps=20000
#opt="Muon"
#for prj in 150
#do
#  for lr_muon in 0.05 #0.05 # 0.005 0.1
#  do
#      # default lr: 8e-4
#      # default num_training_steps 20_000
#      echo ">>> OPTIMIZER: $opt"
#      echo ">>> LR: $lr_muon"
#      torchrun --standalone --nnodes 1 --nproc_per_node 1 torchrun_main.py \
#          --project_name "130m_${opt}_${lr_muon}_gradspikes" \
#          --model_config configs/llama_130m.json \
#          --lr_muon $lr_muon \
#          --density 1.0 \
#          --update_gap 500 \
#          --batch_size 32  \
#          --total_batch_size 512 \
#          --num_training_steps $num_training_steps \
#          --warmup_steps 1000 \
#          --weight_decay 0 \
#          --dtype bfloat16 \
#          --eval_every 1000 \
#          --threshold 5000 \
#          --save_dir $save_dir \
#          --optimizer $opt \
#          --warmup_epoch $prj \
#          --single_gpu
#  done
#done

# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"

