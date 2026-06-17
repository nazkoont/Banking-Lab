#!/bin/bash
#SBATCH --job-name=scrape_deeper
#SBATCH --output=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_2f_%j.log
#SBATCH --error=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_2f_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=snpatel7@stanford.edu

cd /zfs/projects/faculty/nazkoont-baas/fintech_timelines
python3 Code/2f_scrape_deeper.py >> Data_wayback/intermediate_files/deeper_log.txt 2>&1
