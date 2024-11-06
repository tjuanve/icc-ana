#!/bin/bash
set -e
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job is running in directory: "; /bin/pwd

echo "sourcing environment"
source /data/user/tvaneede/software/envs/setup_py3-v4.2.1_nnmfit-v0.3.0.sh
echo `which python`

# see /data/user/tvaneede/software/NNMFit/examples/run_asimov_fit.ipynb 

###
### asimov
###
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/ \
#     -o /data/user/tvaneede/GlobalFit/analysis/toy_km3net/output/example_asimov_test_for_cluster.pickle

# # icecube 10 yr with signal
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/ \
#     -o /data/user/tvaneede/GlobalFit/analysis/toy_km3net/output/example_asimov_test_for_cluster.pickle
    
# # icecube 10 yr, km3net 0.1 year, with signal
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube_km3net.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/ \
#     -o /data/user/tvaneede/GlobalFit/analysis/toy_km3net/output/example_asimov_test_for_cluster.pickle
    


###
### data
###
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/data_SPL_icecube_km3net.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/ \
#     -o /data/user/tvaneede/GlobalFit/analysis/toy_km3net/output/example_data_test.pickle
# Fit result is: (9.942998058904768, {'astro_norm': 0.07723549039306357, 'gamma_astro': 1.541847129928481}, {'success': True, 'message': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'nfev': 72, 'nit': 21, 'warnflag': 0})

python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
    --main_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/main_SPL.cfg \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/analysis_configs/asimov_SPL_icecube_km3net_bestfit.yaml \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/toy_km3net/configs/ \
    -o /data/user/tvaneede/GlobalFit/analysis/toy_km3net/output/example_asimov_test_for_cluster.pickle
    


echo "Job complete!"
