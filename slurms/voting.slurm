#!/bin/bash
#SBATCH --job-name=data_creation_array
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/grid_run.log

# ---- 1. 加载必要模块 ----
source ~/.bashrc
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

# 参数取值
datanames=("aime_2025" "aime_2024" "brumo_2025" "hmmt_2025")
versions=("1" "2")
ifcnts=("True" "False")

# 遍历所有组合
for d in "${datanames[@]}"; do
  for v in "${versions[@]}"; do
    for c in "${ifcnts[@]}"; do
      echo ">>> Running dataname=$d, version=$v, ifcnt=$c at $(date)" >> logs/grid_run.log
      python examples/online_RL/approach_exploration/pool_information/voting.py --dataname "$d" --version "$v" --ifcnt "$c" >> logs/grid_run.log 2>&1
      echo ">>> Finished dataname=$d, version=$v, ifcnt=$c at $(date)" >> logs/grid_run.log
      echo "" >> logs/grid_run.log
    done
  done
done