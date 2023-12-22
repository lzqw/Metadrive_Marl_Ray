
import argparse

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4
from metadrive import MultiAgentIntersectionEnv

from utils.env_wrappers import get_rllib_compatible_env

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)
if __name__ == "__main__":
    args = parser.parse_args()
    # def env_creator(args):
    #
    #     return PettingZooEnv(waterworld_v4.env())
    # env = env_creator({})
    # register_env("waterworld", env_creator)
    # print(env.observation_space_contains)

    from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
    from tqdm import trange
    name,env=get_rllib_compatible_env(MultiAgentIntersectionEnv,True)
    env=env({})

    def my_policy_mapping(agent_id, *args, **kwargs):
        return "test"
    config = (
        PPOConfig()
        .environment(name,env_config={"num_agents":2})
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=1)
        .multi_agent(
            policies={"test"},
            policy_mapping_fn=(my_policy_mapping),
        )
    )

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 60000}


    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space=config,
    ).fit()
