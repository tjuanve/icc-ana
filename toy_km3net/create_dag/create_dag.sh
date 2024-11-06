#!/bin/bash

## 10 year icecube, with signal
# I tried several ranges for astro norm and gamma, final was gamma_astro 1.3 3.9 15 astro_norm 0.01 15.0 15
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/create_dag/output/test_range3_icecube_norm1.36_gamma2.37 \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube.yaml \
#     --main_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --config_dir /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 4.0 \
#     --do_bestfit \
#     --scan gamma_astro 1.3 3.9 15 \
#     --scan astro_norm 0.01 15.0 15

# ## 10 year icecube, 0.1 year km3net, with signal
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/create_dag/output/icecube_km3net_norm1.36_gamma2.37 \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube_km3net.yaml \
#     --main_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --config_dir /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 4.0 \
#     --do_bestfit \
#     --scan gamma_astro 1.3 3.9 15 \
#     --scan astro_norm 0.01 15.0 15

## 10 year icecube, 0.1 year km3net, with signal, based on best fit
# Fit result is: (9.942998058904768, {'astro_norm': 0.07723549039306357, 'gamma_astro': 1.541847129928481}, {'success': True, 'message': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'nfev': 72, 'nit': 21, 'warnflag': 0})
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/create_dag/output/icecube_km3net_norm0.077_gamma1.54 \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube_km3net_bestfit.yaml \
#     --main_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --config_dir /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 4.0 \
#     --do_bestfit \
#     --scan gamma_astro 1.3 3.9 15 \
#     --scan astro_norm 0.01 15.0 15

# lets smaller the range
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/create_dag/output/icecube_km3net_norm0.077_gamma1.54_range2 \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube_km3net_bestfit.yaml \
    --main_config /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
    --config_dir /mnt/ceph1-npx/user/tvaneede/GlobalFit/analysis/toy_km3net/configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 4.0 \
    --do_bestfit \
    --scan gamma_astro 0.5 2.0 15 \
    --scan astro_norm 0.01 7.0 15
