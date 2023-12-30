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

COMM_ACTIONS = "comm_actions"
COMM_PREV_ACTIONS = "comm_prev_actions"

# prev_obs_{t-1} is the concatenation of neighbors' message comm_action_{t-1}
COMM_PREV_OBS = "comm_prev_obs"

# current_obs_t is the concatenation of neighbors' message comm_action_{t-1}
COMM_CURRENT_OBS = "comm_current_obs"
COMM_PREV_2_OBS = "comm_prev_2_obs"

COMM_LOGITS = "comm_logits"
COMM_LOG_PROB = "comm_log_prob"
ENV_PREV_OBS = "env_prev_obs"

COMM_METHOD = "comm_method"

NEI_OBS = "nei_obs"


class CCEnv:

    @classmethod
    def default_config(cls):
        config = super(CCEnv, cls).default_config()
        # Note that this config is set to 40 in LCFEnv
        config["neighbours_distance"] = 40

        config.update(
            dict(
                communication=dict(comm_method="none", comm_size=4, comm_neighbours=4, add_pos_in_comm=False),
                add_traffic_light=False,
                traffic_light_interval=30,
            )
        )

        return config

    def __init__(self, *args, **kwargs):
        #({'num_agents': 2},) {}
        super(CCEnv, self).__init__(*args, **kwargs)
        #print(self.config["communication"]){'comm_method': 'none', 'comm_size': 4, 'comm_neighbours': 4, 'add_pos_in_comm': False}
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        if self.config["communication"][COMM_METHOD] != "none":
            self._comm_obs_buffer = defaultdict()

        if self.config["communication"]["add_pos_in_comm"]:
            self._comm_dim = self.config["communication"]["comm_size"] + 3
        else:
            self._comm_dim = self.config["communication"]["comm_size"]

    def _get_reset_return(self):
        if self.config["communication"][COMM_METHOD] != "none":
            self._comm_obs_buffer = defaultdict()
        obs,i=super(CCEnv, self)._get_reset_return()

        self._update_distance_map()
        # print(self.distance_map)
        for id in i.keys():
            i[id]["cost"] =999
        for kkk in i.keys():
            i[kkk]["all_agents"] = list(i.keys())
            neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
            i[kkk]["neighbours"] = neighbours
            i[kkk]["neighbours_distance"] = nei_distances
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return (obs,i)

    @property
    def action_space(self):
        old_action_space = super(CCEnv, self).action_space
        if not self.config["communication"][COMM_METHOD] != "none":
            return old_action_space
        assert isinstance(old_action_space, Dict)
        new_action_space = Dict[
            {
                k: Box(
                    low=single.low[0],
                    high=single.high[0],
                    dtype=single.dtype,

                    # We are not using self._comm_dim here!
                    shape=(single.shape[0] + self.config["communication"]["comm_size"],)
                )
                for k, single in old_action_space.spaces.items()
            }
        ]
        #Dict('agent0': Box(-1.0, 1.0, (2,), float32), 'agent1': Box(-1.0, 1.0, (2,), float32))
        return new_action_space #Dict('agent0': Box(-1.0, 1.0, (2,), float32), 'agent1': Box(-1.0, 1.0, (2,), float32))

    def step(self, actions):

        if self.config["communication"][COMM_METHOD] != "none":
            comm_actions = {k: v[2:] for k, v in actions.items()}
            actions = {k: v[:2] for k, v in actions.items()}
        o, r, d,t, i = super(CCEnv, self).step(actions)#o, r, tm, tc, info
        self._update_distance_map(dones=d)
        # print(self.distance_map)
        for kkk in i.keys():
            i[kkk]["all_agents"] = list(i.keys())

            neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
            i[kkk]["neighbours"] = neighbours
            i[kkk]["neighbours_distance"] = nei_distances

            if self.config["communication"][COMM_METHOD] != "none":
                i[kkk][COMM_CURRENT_OBS] = []
                for n in neighbours[:self.config["communication"]["comm_neighbours"]]:
                    if n in comm_actions:
                        if self.config["communication"]["add_pos_in_comm"]:
                            ego_vehicle = self.vehicles_including_just_terminated[kkk]
                            nei_vehicle = self.vehicles_including_just_terminated[n]
                            relative_position = ego_vehicle.projection(nei_vehicle.position - ego_vehicle.position)
                            dis = np.linalg.norm(relative_position)
                            extra_comm_obs = [
                                dis / 20, ((relative_position[0] / dis) + 1) / 2, ((relative_position[1] / dis) + 1) / 2
                            ]
                            tmp_comm_obs = np.concatenate([comm_actions[n], np.clip(np.asarray(extra_comm_obs), 0, 1)])
                        else:
                            tmp_comm_obs = comm_actions[n]
                        i[kkk][COMM_CURRENT_OBS].append(tmp_comm_obs)
                    else:
                        i[kkk][COMM_CURRENT_OBS].append(np.zeros((self._comm_dim, )))
        return o, r, d,t, i

    def _find_in_range(self, v_id, distance):
        if distance <= 0:
            return [], []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        ret2 = [
            dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret, ret2

    def _update_distance_map(self, dones=None):
        self.distance_map.clear()
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
            # if dones is not None:
            #     assert (set(dones.keys()) - set(["__all__"])) == set(vehicles.keys()), (dones, vehicles)
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        keys = [k for k, v in vehicles.items() if v is not None]
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = vehicles[k1].position
                p2 = vehicles[k2].position
                distance = np.linalg.norm(p1 - p2)
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance


def get_ccenv(env_class):
    name = env_class.__name__

    class TMP(CCEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


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



def get_rllib_compatible_env_cc(env_class, return_class=False):
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

    name,my_env=get_rllib_compatible_env_cc(get_ccenv(MultiAgentIntersectionEnv), True)
    env=my_env({"num_agents": 2})
    print(env.reset())
    for i in range(10000):
        o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
        if tm["__all__"]:
            print("=============================")
            print(env.reset())
    # print("===============ooooooooooo===================")
    # print(o)
    # print("==========================info=========================")
    # print(info)

    # my_env=get_ccenv(MultiAgentIntersectionEnv)
    # env=my_env({"num_agents": 10,"use_render":True})
    #
    # env.reset()
    # print(env.action_space)
    # o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
    # for i in range(1, 10000000000):
    #     o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})

    # print(env.observation_space)
    # name, env = get_rllib_compatible_env(MultiAgentIntersectionEnv, True)
    # # name2, env2 = get_rllib_compatible_env_origin(MultiAgentIntersectionEnv, True)
    # my_env = env({"num_agents": 2})
    # print(my_env.action_space)
    # print(my_env.observation_space)
    # my_env2 = env2({"num_agents": 2})
    # my_env.reset()
    # print(my_env.observation_space)
    # print(my_env.action_space.sample())
    # obs, info = my_env.reset()
    # print(obs.keys())
    # print(my_env.action_space_sample(list(obs.keys())))
    # o, r, tm, tc, info = my_env.step({agent_id: [0, 0] for agent_id in my_env.vehicles.keys()})
    # print(tm,tc)
    # # o2, r2, tm2, tc2, info2 = my_env2.step({agent_id: [0, 0] for agent_id in my_env2.vehicles.keys()})
    # # print(o2)
