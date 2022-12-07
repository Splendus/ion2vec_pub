
cores = [50, 50, 50, 50, 100, 100, 100, 100]
runtimes = ['01:00:00', '02:00:00', '03:00:00', '06:00:00', '01:00:00', '02:00:00', '03:00:00', '06:00:00']
iters = [1, 2, 5, 10, 1, 2, 5, 10]




for core, runtime, it in zip(cores, runtimes, iters):
    with open(f'p2v_job_{core}cores_{it}iters.sh', 'w') as jobsh:
        jobsh.write(
f'''#!/bin/bash
#SBATCH -A alexandr                               # group to which you belong
#SBATCH -N 1                                      # number of nodes
#SBATCH -n {core}                                 # number of cores
#SBATCH --mem 50G                                 # memory pool for all cores
#SBATCH -t 0-{runtime}                            # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.{core}cores_{it}iters.out        # STDOUT
#SBATCH -e slurm.{core}cores_{it}iters.err        # STDERR
#SBATCH --mail-type=END,FAIL                      # notifications for job done & fail
#SBATCH --mail-user=dominik.geng@embl.de          # send-to address

time python word2vec_pix.py -iter {it}  -threads {core} -train pixel_dataframes -output {core}cores_{it}iters
        ''')
