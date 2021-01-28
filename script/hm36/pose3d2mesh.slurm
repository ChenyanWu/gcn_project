#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-AI
#SBATCH --gres=gpu:volta16:4
#SBATCH --output=/pylon5/ir5fp5p/chenyan/coding/iccv2021/multi_person_mesh_gcn/script/hm36/log/experiment1.out
#SBATCH -t 36:00:00
date
source /pylon5/ir5fp5p/chenyan/opt/anaconda3/etc/profile.d/conda.sh
conda activate open-mmlab

set -x
cd /pylon5/ir5fp5p/chenyan/coding/iccv2021/multi_person_mesh_gcn
time python main/train.py --gpu 0,1,2,3 --cfg ./asset/yaml/pose3d2mesh_human36J_train_human36.yml

date