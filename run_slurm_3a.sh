#!/bin/bash
#SBATCH --job-name=extract_context
#SBATCH --output=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_3a_%j.log
#SBATCH --error=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_3a_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=snpatel7@stanford.edu

cd /zfs/projects/faculty/nazkoont-baas/fintech_timelines
python3 Code/3a_extract_context.py >> Data_wayback/intermediate_files/extract_context_log.txt 2>&1
