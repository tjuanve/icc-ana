!/bin/bash

## track + shower, is the hybrid cascade missing?
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/create_dag/output \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/asimov/asimov_Poisson.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 2.1 2.7 17 \
    --scan astro_norm 1.0 2.0 17

## track
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/create_dag/output_track \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/asimov/asimov_Tracks_Poisson.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 2.1 2.7 17 \
    --scan astro_norm 1.0 2.0 17

### cascade
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/create_dag/output_cscd \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/asimov/asimov_Cscd_Poisson.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 2.1 2.7 17 \
    --scan astro_norm 1.0 2.0 17

## track + cascade, including hybrid
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/create_dag/output_track_cscd \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/asimov/asimov_Tracks_Cscd_Poisson.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 2.1 2.7 17 \
    --scan astro_norm 1.0 2.0 17
