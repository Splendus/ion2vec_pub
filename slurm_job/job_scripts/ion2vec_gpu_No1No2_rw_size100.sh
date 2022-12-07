#!/bin/bash
#SBATCH -p gpu
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 6                                    # number of cores
#SBATCH -C gpu=V100
#SBATCH --gres=gpu:2
#SBATCH --mem 300G                                # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ion2vec_TheoNo1No2_TF_rw_size100.out   # STDOT
#SBATCH -e slurm.ion2vec_TheoNo1No2_TF_rw_size100.err   # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python ion2vec_TF.py -output TheoNo1No2_TF_rw_size100 -train slurm_job/theos_recom/No1No2 -quan 0 -pix_per 1. -size 100 -rw 1
