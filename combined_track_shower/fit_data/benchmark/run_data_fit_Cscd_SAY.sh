#!/bin/bash
set -e
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job is running in directory: "; /bin/pwd

echo "sourcing environment"
source /data/user/tvaneede/software/envs/setup_py3-v4.2.1_nnmfit-v0.3.0.sh
echo `which python`

python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
    --main_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/main.cfg \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs/analysis_configs/data/data_Cscd_SAY.yaml \
    --config_dir /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/GlobalFit_configs \
    -o /data/user/tvaneede/GlobalFit/analysis/combined_track_shower/output/data_Cscd_SAY.pickle

echo "Job complete!"
