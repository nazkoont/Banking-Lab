#!/bin/bash
#SBATCH --job-name=extract_llm
#SBATCH --output=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_3b_%j.log
#SBATCH --error=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_3b_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=snpatel7@stanford.edu

cd /zfs/projects/faculty/nazkoont-baas/fintech_timelines
python3 Code/3b_extract_product_info.py >> Data_wayback/intermediate_files/extract_log.txt 2>&1
