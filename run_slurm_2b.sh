#!/bin/bash
#SBATCH --job-name=wayback_2b
#SBATCH --output=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_2b_%j.log
#SBATCH --error=/zfs/projects/faculty/nazkoont-baas/fintech_timelines/Data_wayback/intermediate_files/slurm_2b_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=snpatel7@stanford.edu

cd /zfs/projects/faculty/nazkoont-baas/fintech_timelines
python3 Code/2b_find_product_urls.py >> Data_wayback/intermediate_files/find_urls_log.txt 2>&1
