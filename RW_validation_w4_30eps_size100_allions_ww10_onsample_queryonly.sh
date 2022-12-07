#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 12                                    # number of cores
#SBATCH --mem 100G                                # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.RW_validation_w4_30eps_size100_allions_ww10_onsample_queryonly.out   # STDOT
#SBATCH -e slurm.RW_validation_w4_30eps_size100_allions_ww10_onsample_queryonly.err   # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python rw_train.py -ind_name train_ions.pickle -output RW_validation_w4_30eps_size100_allions_ww10_onsample_queryonly -train datasets/theos_recom/mouse_wb_pos -quan 0 -pix_per 1 -word_window 10 -window 4 -size 100 -iter 30 -rw 1 