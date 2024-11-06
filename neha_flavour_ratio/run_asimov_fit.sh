#!/bin/bash
set -e
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job is running in directory: "; /bin/pwd

echo "sourcing environment"
source /data/user/tvaneede/software/envs/setup_py3-v4.2.1_nnmfit-v0.3.0.sh
echo `which python`

###
### asimov_Poisson_TrackBestFit
###
python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
    --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/Poisson/asimov_Poisson_TrackBestFit.yaml \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
    -o /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/testfit_output/asimov_Poisson_TrackBestFit/asimov_Poisson_TrackBestFit.pickle

# ###
# ### asimov_Poisson_HESEBestfit
# ###
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/Poisson/asimov_Poisson_HESEBestfit.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     -o /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/testfit_output/asimov_Poisson_HESEBestfit/asimov_Poisson_HESEBestfit.yaml

# ###
# ### asimov_Poisson_HESE12Bestfit
# ###
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/Poisson/asimov_Poisson_HESE12Bestfit.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     -o /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/testfit_output/asimov_Poisson_HESE12Bestfit/asimov_Poisson_HESE12Bestfit.yaml

# ###
# ### asimov_Poisson_GFSPL
# ###
# python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/Poisson/asimov_Poisson_GFSPL.yaml \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     -o /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/testfit_output/asimov_Poisson_GFSPL/asimov_Poisson_GFSPL.yaml


echo "Job complete!"
