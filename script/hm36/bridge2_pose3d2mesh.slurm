#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 37:00:00
#SBATCH --gpus=4
#SBATCH --output=/ocean/projects/iri180005p/chenyan/coding/gcn_project/script/hm36/log/experiment_bridge2.out

date
source /ocean/projects/iri180005p/chenyan/opt/anaconda3/etc/profile.d/conda.sh
conda activate py38

set -x
cd /ocean/projects/iri180005p/chenyan/coding/gcn_project
time python main/train.py --gpu 0,1,2,3 --cfg ./asset/yaml/pose3d2mesh_human36J_train_human36.yml

date