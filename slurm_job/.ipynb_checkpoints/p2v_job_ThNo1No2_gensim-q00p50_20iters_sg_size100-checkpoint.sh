#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 64                                    # number of cores
#SBATCH --mem 100G                                 # memory pool for all cores
#SBATCH -t 3-00:00:00                             # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ThNo1No2_q00p50_20iters_sg_size100.out         # STDOUT
#SBATCH -e slurm.ThNo1No2_q00p50_20iters_sg_size100.err         # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python word2vec_pix.py -iter 20 -threads 64 -train theos_recom/No1No2 -output ThNo1No2gensim_q00p50_20iters_sg_size100 -size 100 -pix_per 0.5 -stride 4 -quan 0 -cbow 0
        
