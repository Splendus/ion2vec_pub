#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 100                                 # number of cores
#SBATCH --mem 50G                                 # memory pool for all cores
#SBATCH -t 3-00:00:00                            # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ThNo1.out        # STDOUT
#SBATCH -e slurm.ThNo1.err        # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python word2vec_pix.py -iter 10  -threads 100 -train theos_recom/No1 -output ThNo1_gensim -size 20 -int_per 0
        
