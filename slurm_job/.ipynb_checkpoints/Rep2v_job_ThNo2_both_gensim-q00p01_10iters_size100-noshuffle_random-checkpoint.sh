#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 64                                    # number of cores
#SBATCH --mem 50G                                 # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ThNo2_both_q00p01_10iters_cbow_noshuffle100randomRerun.out         # STDOUT
#SBATCH -e slurm.ThNo2_both_q00p01_10iters_cbow_noshuffle100randomRerun.err         # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python word2vec_pix.py -iter 10 -threads 64 -train theos_recom/No2_pos_neg -output ThNo2_both_gensim_q00p01_10iters_noshuffle100randomRerun -size 100 -pix_per 0.1 -quan 0 -cbow 1 -shuffle 0
        
