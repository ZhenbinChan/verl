srun -p gpu_a800 -G1 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/one_epoch_dapo.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G1 --exclude=d1n41a22g03 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/format_step_gdpo.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G2 --exclude=d1n41a25g03 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/self_eval_step_gdpo_local.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G1 --exclude=d1n41a23g02 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/sanity_check_tree_gae.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G1 bash -c "ray stop --force; export PYTHONUNBUFFERED=1; bash bash_scripts/self_eval_step_gdpo_1gpu.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G2 --exclude=d1n41a25g03 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/self_eval_tree_gae_local.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G2 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/fol_step_gdpo_local.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G2 bash -c "ray stop; export PYTHONUNBUFFERED=1; bash bash_scripts/sanity_check_fol_step_gdpo.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
srun -p gpu_a800 -G1 bash -c "ray stop; export PYTHONUNBUFFERED=1; bash bash_scripts/fol_step_gdpo_remote.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log

d1n41a25g03