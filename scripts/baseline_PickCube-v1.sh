# rgb based baselines
seed=1
for demos in 100; do
  python3 train_rgb.py --env-id PickCube-v1 \
    --demo-path /data/PickCube-v1/motionplanning/trajectory.h5 \
    --control-mode "pd_joint_pos" \
    --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
    --total_iters 30000 --obs-mode "rgb" \
    --exp-name diffusion_policy-PickCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track
done