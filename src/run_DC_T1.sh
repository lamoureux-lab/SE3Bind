#!/bin/bash

### train


###SBATCH --job-name=SE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=EVALSE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign
#SBATCH --job-name=EVal1000_crystalSE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign

###SBATCH --job-name=SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=EVALSE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=RES910SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=EVAL1000_crystalSE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign

###SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2vn
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2vn

##SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v
##SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v

###SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign
##SBATCH --job-name=RES1000_SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=RES1450_SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign

###SBATCH --job-name=plotFeatures_crystal_evalepoch1500_SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign
##SBATCH --job-name=plotFeatures_Modeller_evalepoch1500_SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign

##SBATCH --job-name=SE3Bind_exp48_B_JT_f0lossplusminus_4L3s2v_200ep_asign

###SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp40_B_JT_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep_asign
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp40_B_JT_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep_asign

###SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep

###SBATCH --job-name=RES3010_SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2vn
###SBATCH --job-name=EVAL3000_Crystal_SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2vn
##SBATCH --job-name=EVAL3000_Modeller_SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2vn

###SBATCH --job-name=check500_SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign
###SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign

##SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp34_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_3L3s2v_200ep_asign
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp34_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_3L3s2v_200ep_asign

###SBATCH --job-name=RES1030_SE3Bind_exp45_B_absf0_L1loss_3L3s2v
###SBATCH --job-name=EVAL1000_Crystal_SE3Bind_exp45_B_absf0_L1loss_3L3s2v
###SBATCH --job-name=EVAL1000_Modeller_SE3Bind_exp45_B_absf0_L1loss_3L3s2v


###SBATCH --job-name=RES1030_SE3Bind_exp46_B_noE0_L1loss_3L3s2v
###BATCH --job-name=EVAL1000_Modeller_SE3Bind_exp46_B_noE0_L1loss_3L3s2v
###SBATCH --job-name=EVAL1000_Crystal_SE3Bind_exp46_B_noE0_L1loss_3L3s2v
###SBATCH --job-name=RES710_SE3Bind_exp47_E0plusminusF0_L1loss_3L3s2v

###SBATCH --job-name=GETmean_GT_deltaGs_forlogfile

##SBATCH --job-name=EVALevery10_Modeller_SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign
###SBATCH --job-name=EVALevery10_Crystal_SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign

### run on cgpu newark
#sbatch --cluster=amareln
#SBATCH --partition=gpu              # Partition (job queue)
##SBATCH --exclude=gpun003


### run on cgpu camden
##sbatch --cluster=amarelc
##SBATCH --partition=cgpu              # Partition (job queue)
##SBATCH --constraint=ampere&oarc

### run on cgpu piscataway
##sbatch --cluster=amarel
###SBATCH --partition=gpu              # Partition (job queue)
##SBATCH --constraint=ampere

#SBATCH --nodes=1                   # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=12            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                  # Number of GPUs
#SBATCH --mem=72G                 # Real memory (RAM) required (MB), 0 is the whole-node memory
###SBATCH --time=15:00:00           # Total run time limit (HH:MM:SS)
###SBATCH --time=10:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --time=3-00:00:00           # Total run time limit (HH:MM:SS)
###SBATCH --time=17:00:00           # Total run time limit (HH:MM:SS)

###SBATCH --constraint=volta
###SBATCH --constraint=pascal

### tesla
###SBATCH --cpus-per-task=12
###SBATCH --constraint=oarc
###SBATCH --mem=64000
###SBATCH --time=3-00:00:00

### pascal
##BATCH --cpus-per-task=32
##SBATCH --constraint=oarc
##SBATCH --mem=64GB
##SBATCH --time=3-00:00:00

### volta
##SBATCH --cpus-per-task=40
##SBATCH --constraint=volta
##SBATCH --mem=192000
##SBATCH --mem=64GB
##SBATCH --time=3-00:00:00

### all other gpus
##SBATCH --partition=gpu              # Partition (job queue)
##SBATCH --ntasks=1                   # Total # of tasks across all nodes
##SBATCH --gres=gpu:2                 # Number of GPUs
##SBATCH --cpus-per-task=12
######SBATCH --mem=192000
##SBATCH --mem=64GB
##SBATCH --time=3-00:00:00
####SBATCH --nodelist=gpu006   GPU006 fast!


### titan ###
###SBATCH --cpus-per-task=24
###SBATCH --constraint=titan
###SBATCH --mem=190000
###SBATCH --time=3-00:00:00

#################################################
##SBATCH --partition=p_ccib_1
###SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1 # Number of GPUs
##SBATCH --constraint=volta
##SBATCH --mem=30G
##SBATCH --time=3-00:00:00
#################################################

### more shared flags
##mkdir -p slurm_log/;
#SBATCH --output=slurm_log_shriya/slurm.%N.%j.%x.out
#SBATCH --error=slurm_log_shriya/slurm.%N.%j.%x.err
#SBATCH --export=ALL

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=as3190@scarletmail.rutgers.edu

#nvidia-smi
echo "Running script..."
mkdir -p slurm_log_shriya/
# Add the parent directory of models to the Python path
export PYTHONPATH="${PYTHONPATH}:../"
pwd
srun -N1 -n1 python train_DC_T1.py;
