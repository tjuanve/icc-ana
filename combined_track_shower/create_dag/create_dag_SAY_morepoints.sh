
###
### say ll
###

### track + shower, no hybrid
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/create_dag/output/SAY/track_cscd_nohybrid \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/asimov/asimov_SAY.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 1.8 2.8 15 \
    --scan astro_norm 0.2 2.3 15


## track
generate_dagman_llhscan.py \
    --output_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/create_dag/output/SAY/track \
    --dag_dir /scratch/tvaneede/NNMFit/condor \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/asimov/asimov_Tracks_SAY.yaml \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    --virtualenv /data/user/tvaneede/software/py_venvs/py3-v4.2.1_nnmfit-v0.3.0 \
    --scan_memory 8.0 \
    --do_bestfit \
    --scan gamma_astro 1.8 2.8 15 \
    --scan astro_norm 0.2 2.3 15

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
    --scan gamma_astro 1.8 2.8 15 \
    --scan astro_norm 0.2 2.3 15
