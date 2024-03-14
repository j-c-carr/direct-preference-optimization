#!/bin/bash
#SBATCH --output=job_output_pythia28_sft.txt
#SBATCH --error=job_error_pythia28_sft.txt
#SBATCH --mem=100Gb
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:a100l:4
#SBATCH --constraint="ampere"
#SBATCH --time=2:30:00
#SBATCH --mail-user=jonathan.colaco-carr@mila.quebec


module load python/3.8
module load cuda/11.7

# Location of the DPO repository and virtual environment on the mila cluster
dpo_dir=$HOME/courses/c597/direct-preference-optimization

# Activate the virtual environment
source $dpo_dir/venv/bin/activate

# Perform SFT
python -u $dpo_dir/train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

