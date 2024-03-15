#!/bin/bash
#SBATCH --output=job_output_pythia28_dpo.txt
#SBATCH --error=job_error_pythia28_dpo.txt
#SBATCH --mem=120Gb
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:a100l:4
#SBATCH --constraint="ampere"
#SBATCH --time=3:30:00
#SBATCH --mail-user=jonathan.colaco-carr@mila.quebec


module load python/3.8
module load cuda/11.7

# Location of the DPO repository
dpo_dir=$HOME/courses/c597/direct-preference-optimization

# Activate the virtual environment
source $dpo_dir/venv/bin/activate


# Pythia 2.8B SFT Checkpoint, created after SFT has completed
pythia_sft_checkpoint=$HOME/scratch/.cache/jonathan.colaco-carr/anthropic_dpo_pythia28_2024-02-29_20-49-33_380098/LATEST/policy.pt

# Run DPO with Pythia 2.8B
python -u $dpo_dir/train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=$pythia_sft_checkpoint
