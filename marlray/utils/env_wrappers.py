import copy
import logging
import random
from collections import defaultdict
from math import cos, sin
from typing import Optional, Tuple, Dict, Any

import numpy as np
from gymnasium.spaces import Box, Dict
from metadrive.utils import get_np_random, clip
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env


def get_rllib_compatible_env_origin(env_class, return_class=False):
    env_name = env_class.__name__

    class MA(env_class, MultiAgentEnv):
        _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, config, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
            MultiAgentEnv.__init__(self)

        @property
        def observation_space(self):
            ret = super(MA, self).observation_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        @property
        def action_space(self):
            ret = super(MA, self).action_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        def action_space_sample(self, agent_ids: list = None):
            """
            RLLib always has unnecessary stupid requirements that you have to bypass them by overwriting some
            useless functions.
            """
            return self.action_space.sample()

    MA.__name__ = env_name
    MA.__qualname__ = env_name
    register_env(env_name, lambda config: MA(config))

    if return_class:
        return env_name, MA

    return env_name


def get_rllib_compatible_env(env_class, return_class=False):
    env_name = env_class.__name__

    class MA(env_class, MultiAgentEnv):
        _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, config, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
            MultiAgentEnv.__init__(self)

        @property
        def observation_space(self):
            ret = super(MA, self).observation_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            id = random.sample(list(ret.keys()), 1)

            return ret[id[0]]

        @property
        def action_space(self):
            ret = super(MA, self).action_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            id = random.sample(list(ret.keys()), 1)
            return ret[id[0]]

        def action_space_sample(self, agent_ids: list = None):
            # {0: 1, 1: 0, 2: 1, 3: 1}
            return {agent_id: self.action_space.sample() for agent_id in agent_ids}

        def reset(self,
                  *,
                  seed: Optional[int] = None,
                  options: Optional[dict] = None, ):
            return env_class.reset(self, seed=0)

    class MA_gymnasium(MA):
        def __init__(self, config, *args, **kwargs):
            MA.__init__(self, config, *args, **kwargs)

        def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict] = None,
        ) -> Tuple[MultiAgentDict, MultiAgentDict]:
            # self.logger.setLevel(logging.FATAL)
            return super().reset(seed=seed, options=options)

        def step(self, action_dict: MultiAgentDict) -> tuple[
            dict[Any, Any], dict[Any, Any], dict[Any, Any], dict[Any, Any], dict[Any, Any]]:
            o, r, tm, tc, info = super().step(action_dict)
            tc['__all__'] = all(tc.values())

            return o, r, tm, tc, info

    MA_gymnasium.__name__ = env_name
    MA_gymnasium.__qualname__ = env_name
    register_env(env_name, lambda config: MA_gymnasium(config))

    if return_class:
        return env_name, MA_gymnasium

    return env_name


if __name__ == '__main__':
    # Test if the distance map is correctly updated.
    from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
    from tqdm import trange

    name, env = get_rllib_compatible_env(MultiAgentIntersectionEnv, True)
    name2, env2 = get_rllib_compatible_env_origin(MultiAgentIntersectionEnv, True)
    my_env = env({"num_agents": 3})
    my_env2 = env2({"num_agents": 2})
    # my_env.reset()
    print(my_env.observation_space)
    print(my_env.action_space.sample())
    obs, info = my_env.reset()
    print(obs.keys())
    print(my_env.action_space_sample(list(obs.keys())))
    o, r, tm, tc, info = my_env.step({agent_id: [0, 0] for agent_id in my_env.vehicles.keys()})
    print(tm,tc)
    # o2, r2, tm2, tc2, info2 = my_env2.step({agent_id: [0, 0] for agent_id in my_env2.vehicles.keys()})
    # print(o2)
