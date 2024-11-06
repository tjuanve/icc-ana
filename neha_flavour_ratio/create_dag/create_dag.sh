#!/bin/bash

###
### asimov_Poisson_TrackBestFit
###
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_Poisson_TrackBestFit/diffuse \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/Poisson/asimov_Poisson_TrackBestFit.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --scan gamma_astro 2.1 2.7 10 \
#     --scan astro_norm 1.0 1.9 10

# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_Poisson_TrackBestFit/flavor \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/Poisson/asimov_Poisson_TrackBestFit.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --scan astro_nue_ratio 0.0 3.0 10 \
#     --scan astro_nutau_ratio 0.0 3.0 10

# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_SAYLLH_TrackBestFit/flavor \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/SAY/asimov_SAY_TrackBestFit.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --scan astro_nue_ratio 0.0 3.0 10 \
#     --scan astro_nutau_ratio 0.0 3.0 10

# ### more measurement points
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_SAYLLH_TrackBestFit_largerRange/flavor \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/SAY/asimov_SAY_TrackBestFit.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --scan astro_nue_ratio 0.0 6.0 10 \
#     --scan astro_nutau_ratio 0.0 6.0 10

# ## custom range
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_SAYLLH_TrackBestFit_customRange/flavor \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/SAY/asimov_SAY_TrackBestFit.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --custom_scan_points /home/rnaab/analysis/GlobalFit_analysis/results_step3/combined/astro_models/default_custom_scan_points_flavor.pickle

## turn off the systematics
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_SAYLLH_TrackBestFit_nosyst/flavor \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/SAY/asimov_SAY_TrackBestFit.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --custom_scan_points /data/user/tvaneede/GlobalFit/custom_scan_flavor/default_custom_scan_points_flavor.pickle \
    --override_configs /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/override/Systematics/NoSystematics.cfg


# ###
# ### asimov_SAY_HESE12Bestfit
# ###
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_SAYLLH_HESE12Bestfit/flavor \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/SAY/asimov_SAY_HESE12Bestfit.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --custom_scan_points /data/user/tvaneede/GlobalFit/custom_scan_flavor/default_custom_scan_points_flavor.pickle


###
### asimov_SAY_HESE12Bestfit_nominalDetSyst
###
# generate_dagman_llhscan.py \
#     --output_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/create_dag/output/asimov_SAYLLH_HESE12Bestfit_nominalDetSyst/flavor \
#     --dag_dir /scratch/tvaneede/NNMFit/condor \
#     --analysis_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/analysis_configs/asimov/SAY/asimov_SAY_HESE12Bestfit_nominalDetSyst.yaml \
#     --main_config /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs/main.cfg \
#     --config_dir /data/user/tvaneede/GlobalFit/analysis/neha_flavour_ratio/NNMFit_Configs \
#     --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
#     --scan_memory 8.0 \
#     --do_bestfit \
#     --custom_scan_points /data/user/tvaneede/GlobalFit/custom_scan_flavor/default_custom_scan_points_flavor.pickle
