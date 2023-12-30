from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

from marlray.algo.mappo import MAPPOTrainer, get_mappo_env
from marlray.utils.callbacks import MultiAgentDrivingCallbacks
from marlray.utils.train import train
from marlray.utils.utils import get_train_parser

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"
    stop = int(100_0000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_mappo_env(MultiAgentIntersectionEnv),
        env_config={"num_agents": 20},
        num_gpus=1 if args.num_gpus != 0 else 0,

        # ===== MAPPO =====
        counterfactual=True,
        fuse_mode="mf",
        mf_nei_distance=10,
    )
    train(
        MAPPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=3,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,

        # fail_fast='raise',
        # local_mode=True
    )


