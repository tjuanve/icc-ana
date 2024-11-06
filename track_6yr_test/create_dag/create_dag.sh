#!/bin/bash

generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/create_dag/output_mem8 \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config  /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/configs/analysis_configs/asimov_SPL.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/configs/main_SPL.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/configs/ \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 2.1 2.7 17 \
    --scan astro_norm 0.8 1.8 17

#     --override_configs resources/configs/override/SnowStorm_DomEff_IceAbs_IceScat_HoleIceP0.cfg resources/configs/override/SnowStorm_GaussWidth0.06.cfg \
