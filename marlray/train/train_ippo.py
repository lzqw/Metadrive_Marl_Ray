import os
import sys

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from ray import tune
sys.path.insert(0, sys.path[0]+"/../")
from marlray.algo.ippo import IPPOTrainer
from marlray.utils.callbacks import MultiAgentDrivingCallbacks
from marlray.utils.env_wrappers  import get_rllib_compatible_env
from marlray.utils.train import train
from marlray.utils.utils import get_train_parser




if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    stop = int(10_000_000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_rllib_compatible_env(MultiAgentIntersectionEnv),
        env_config={
            "num_agents":args.num_agents,
        },
        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=args.num_gpus if args.num_gpus != 0 else 0,
        vf_clip_param=20,
        train_batch_size=args.train_batch_size,
        num_rollout_workers=args.workers,
    )



    # Launch training
    train(
        IPPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,
        checkpoint_freq=50,
        local_mode=False
    )