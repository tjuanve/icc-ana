#!/bin/bash
set -e
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job is running in directory: "; /bin/pwd

echo "sourcing environment"
source /data/user/tvaneede/software/envs/setup_py3-v4.2.1_nnmfit-v0.3.0.sh
echo `which python`

# see /data/user/tvaneede/software/NNMFit/examples/run_asimov_fit.ipynb 
python /mnt/ceph1-npx/user/tvaneede/software/NNMFit/NNMFit/scripts/run_fit.py \
    --main_config /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/configs/main_SPL.cfg \
    --analysis_config /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/configs/analysis_configs/asimov_SPL.yaml --config_dir /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/configs/ \
    -o /data/user/tvaneede/GlobalFit/analysis/track_6yr_test/output/example_asimov_test_for_cluster.pickle

echo "Job complete!"
