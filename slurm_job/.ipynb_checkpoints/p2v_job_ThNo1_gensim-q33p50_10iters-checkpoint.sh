#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 100                                    # number of cores
#SBATCH --mem 50G                                 # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ThNo1_q33p50_10iters.out         # STDOUT
#SBATCH -e slurm.ThNo1_q33p50_10iters.err         # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python word2vec_pix.py -iter 10  -threads 64 -train theos_recom/No1 -output ThNo1_gensim_q33p50_10iters -size 20 -pix_per 0.5 -stride 4 -quan 33.
        
