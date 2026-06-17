#!/bin/bash
#SBATCH --job-name=wayback_missing
#SBATCH --output=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_missing_%j.log
#SBATCH --error=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_missing_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=snpatel7@stanford.edu

export PATH=/opt/jupyterhub/bin:$PATH
cd /zfs/projects/faculty/nazkoont-baas/fintech_timelines
bash Code/2_run_wayback_pipeline.sh --start 2 >> Data_wayback/intermediate_files/pipeline_log_missing.txt 2>&1

# Restore original master after run
cp Data_cleaned/fintech_timelines_master_BACKUP.csv Data_cleaned/fintech_timelines_master.csv
