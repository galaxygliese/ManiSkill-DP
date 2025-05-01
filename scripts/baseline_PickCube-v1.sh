# rgb based baselines
for demos in 100; do
  python3 train_rgbd.py --env-id PickCube-v1 \
    --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
    --total_iters 30000 --obs-mode "rgb" \
    --exp-name diffusion_policy-PickCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track
done