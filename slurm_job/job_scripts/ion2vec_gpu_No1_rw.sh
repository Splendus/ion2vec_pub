#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n 50                                     # number of cores
#SBATCH --mem 300G                                 # memory pool for all cores
#SBATCH -t 3-00:00:00                            # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.ion2vec_TheoNo1_nhalf_rw.out                  # STDOT
#SBATCH -e slurm.ion2vec_TheoNo1_nhalf_rw.err                  # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python ion2vec_rw.py -output TheoNo1_nhalf_rw -train slurm_job/theos_recom/No1
