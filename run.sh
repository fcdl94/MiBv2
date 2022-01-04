#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp='python -m torch.distributed.launch --nproc_per_node=2 --master_port ${port} run.py --num_workers 8 --sample_num 8'
shopt -s expand_aliases
overlap=$2

dataset=voc
epochs=30
task=19-1
lr_init=0.01
lr=0.001

if [ ${overlap} -eq 0 ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
else
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
fi

dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 --epochs 30 $ov --val_interval 2 --random_seed 1"
exp --name FT  --step 0 --lr ${lr_init} ${dataset_pars}
exp --name FT_bce  --step 0 --lr ${lr_init} ${dataset_pars} --bce

#exp --name FT --step 1 ${dataset_pars} --lr ${lr}
#exp --name LWF --method LWF --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT
#exp --name ILT --method ILT --step 1 ${dataset_pars} --lr ${lr} --step_ckpt $pretr_FT
exp --name MiB --method MiB --step 1 ${dataset_pars} --lr ${lr} --step_ckpt ${path}FT_0.pth
exp --name PLOP --method PLOP --step 1 ${dataset_pars} --lr ${lr} --step_ckpt ${path}FT_0.pth
exp --name IC --method IC --step 1 ${dataset_pars} --lr ${lr} --step_ckpt ${path}FT_bce_0.pth


#for i in 2 3 4 5; do
##  exp --name LWF --method LWF --step $i ${dataset_pars} --lr ${lr}
##  exp --name ILT --method ILT --step $i ${dataset_pars} --lr ${lr}
#  exp --name MiB --method MiB --step $i ${dataset_pars} --lr ${lr}
#  exp --name PLOP --method PLOP --step $i ${dataset_pars} --lr ${lr}
#done
