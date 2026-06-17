#!/bin/bash
#SBATCH --job-name=wayback_scrape
#SBATCH --output=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_%j.log
#SBATCH --error=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=snpatel7@stanford.edu

export PATH=/opt/jupyterhub/bin:$PATH
cd /zfs/projects/faculty/nazkoont-baas/fintech_timelines
bash Code/2_run_wayback_pipeline.sh  >> Data_wayback/intermediate_files/pipeline_log.txt 2>&1
