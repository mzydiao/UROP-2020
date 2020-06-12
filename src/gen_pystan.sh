#!/bin/bash

source /broad/software/scripts/useuse
reuse -q Anaconda3

#$ -cwd
#$ -V
#$ -o ./output/
#$ -e error.txt
#$ -l h_vmem=128G
#$ -R y
#$ -j y
#$ -l h_rt=24:00:00

source activate /broad/thechenlab/mdiao/lda-env

##################
### Run script ###
##################

python gen_pystan.py
