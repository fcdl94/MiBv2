#!/bin/bash

#export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp='python -m torch.distributed.launch --nproc_per_node=2 --master_port ${port} run.py --num_workers 8 --sample_num 8'
shopt -s expand_aliases
overlap=$1

dataset=voc
epochs=30
task=15-5
lr_init=0.01
lr=0.001

if [ ${overlap} -eq 0 ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
else
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
fi

pretr_FT=${path}FT_0.pth
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 --epochs 30 $ov --val_interval 5"
#exp --name FT  --step 0 --lr ${lr_init} ${dataset_pars}

#exp --name FT --step 1 ${dataset_pars} --lr ${lr}
#exp --name LWF --method LWF --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT
#exp --name ILT --method ILT --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT
#exp --name MiB_After_plop --method MiB --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT
#exp --name PLOP_NoAMP --method PLOP --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT
exp --name PLOP_NoAMP --method PLOP --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT --test --continue_ckpt

#for i in 2 3 4 5; do
#  exp --name LWF --method LWF --step $i ${dataset_pars} --lr ${lr}
#  exp --name ILT --method ILT --step $i ${dataset_pars} --lr ${lr}
#  exp --name MIB --method MiB --step $i ${dataset_pars} --lr ${lr}