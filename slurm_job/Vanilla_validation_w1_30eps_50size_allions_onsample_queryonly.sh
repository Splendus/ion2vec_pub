#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 32                                    # number of cores
#SBATCH --mem 100G                                 # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.Vanilla_validation-w1_30eps_size50allions_onsample_queryonly.out         # STDOUT
#SBATCH -e slurm.Vanilla_validation-w1_30eps_size50allions_onsample_queryonly.err         # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python word2vec_pix.py -ind_name train_ions.pickle -iter 30 -threads 64 -train theos_recom/mouse_wb_pos -output Vanilla_validation_w1_30size50allions_onsample_queryonly -size 50 -pix_per 1. -quan 0 -window 1
        
