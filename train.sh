export PATH="/mnt/lustre/xuyan2/opt/cuda-9.0/bin:$PATH"; 
export LD_LIBRARY_PATH="/mnt/lustre/xuyan2/opt/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/mnt/lustre/xuyan2/Projects/Works/PointsOdometry:$PYTHONPATH"

mkdir LOG
now=$(date +"%Y%m%d_%H%M%S")
train_file=/mnt/lustre/xuyan2/Projects/Works/spconv_plus_/setup.py
model_dir=./Rslts_$0

srun --partition=AD  --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
     --job-name=vot  --kill-on-bad-exit=1 \
     python -u  $train_file bdist_wheel \
 2>&1 | tee LOG/$0_$now.log &
 	# --gen_type 'resnetG' \
