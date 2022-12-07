#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 64                                    # number of cores
#SBATCH --mem 1000G                                # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ion2vec_TheoNo1No2_TF_Vanilla.out   # STDOT
#SBATCH -e slurm.ion2vec_TheoNo1No2_TF_Vanilla.err   # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python ion2vec.py -output TheoNo1No2_TF_Vanilla -train slurm_job/theos_recom/No1No2 -quan 10 -pix_per 0.5