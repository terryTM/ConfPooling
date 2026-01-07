#!/bin/bash
#SBATCH --job-name=pooling_N_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
# ---- 修改点 1: 数组范围改为 0-6，对应 7 个特定的 N 值 ----
#SBATCH --array=0-6
#SBATCH --output=logs/Ninit_%A_%a.log

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate


# ---- 修改点 2: 定义 N 的取值列表并提取当前值 ----
N_VALUES=(16 32 64 96 128 160 256)
CURRENT_N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running task with index $SLURM_ARRAY_TASK_ID, Ninit = $CURRENT_N"

# ---- 4. 运行程序 ----
# 这里将 $CURRENT_N 传给 --Ninit
python /home/yz54720/Projects/Method/deepconf/examples/online_RL/approach_exploration/pool_information/pooling_package_ws_plot.py \
  --Ninit $CURRENT_N