#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --account=PAS2138
#SBATCH -p gpu
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=1
#SBATCH --mem=128G

module load cuda/11.8.0
source /fs/scratch/PAS1957/delijingyic/miniconda3/bin/activate
conda activate melostscardinal

echo $SLURM_JOB_ID
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_NTASKS

mkdir -p output/$SLURM_JOB_ID
WORK_DIR=output/$SLURM_JOB_ID

set -ex

CONFIG=$1
GPUS=$2
# NODE=$3
# NODE_RANK=$4

MODEL_NAME=$(basename "$(dirname $CONFIG)")
CHECKPOINT_DIR="/home/ubuntu/OpenVoice/emotion_STS/melo_sts/logs_base"

PORT=10908

# torchrun --nproc_per_node=$GPUS \
#         --master_port=$PORT \
#     /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/train.py --c $CONFIG --model $MODEL_NAME --pretrain_rl /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/pretrained_RL
# rm -rf /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/buffer/*
while : # auto-resume: the code sometimes crash due to bug of gloo on some gpus
do

# torchrun --nproc_per_node=$GPUS \
#         --master_port=$PORT \
#     /users/PAS2062/delijingyic/project/OpenVoice/emotion_STS/melo_sts/train.py --c $CONFIG --model $MODEL_NAME  --pretrain_G $CHECKPOINT_DIR --pretrain_D $CHECKPOINT_DIR --pretrain_dur $CHECKPOINT_DIR 
torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
    /users/PAS2062/delijingyic/project/OpenVoice/emotion_STS/melo_sts9/train.py --c $CONFIG --model $MODEL_NAME  

for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}')
do
    echo $PID
    kill -9 $PID
done
sleep 30
done