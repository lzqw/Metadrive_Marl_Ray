import argparse
from metadrive.envs.marl_envs import MultiAgentBottleneckEnv, MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, \
    MultiAgentParkingLotEnv, MultiAgentTollgateEnv
from marlray.algo.ippo import IPPOTrainer
from marlray.utils.env_wrappers import get_rllib_compatible_env


def get_env(env_name, use_native_render=False):
    config = {"use_render": use_native_render}
    if env_name == "inter":
        return MultiAgentIntersectionEnv(config)
    elif env_name == "round":
        return MultiAgentRoundaboutEnv(config)
    elif env_name == "parking":
        return MultiAgentParkingLotEnv(config)
    elif env_name == "tollgate":
        return MultiAgentTollgateEnv(config)
    elif env_name == "bottle":
        return MultiAgentBottleneckEnv(config)
    else:
        raise ValueError("Unknown environment {}!".format(env_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="inter", type=str)
    parser.add_argument("--use_native_render", type=bool, default=False)
    parser.add_argument("--use_3d_render", action="store_true")
    args = parser.parse_args()
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_rllib_compatible_env(MultiAgentIntersectionEnv),
        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0,
        vf_clip_param=20
    )
    policy_function=IPPOTrainer(config)
    policy_function.restore('/home/lzqw/PycharmProjects/Metadrive_Marl_Ray/marlray/ray_results/Intersection/checkpoint_000003')
    # ===== Create environment =====
    _,env = get_rllib_compatible_env(MultiAgentIntersectionEnv,True)
    env= env({"use_render": args.use_native_render,
              "num_agents": 10,
              })
    o,info = env.reset()
    d = {"__all__": False}
    ep_success = 0
    ep_step = 0
    ep_agent = 0

    for i in range(1, 100000):
        o, r, d,truncate, info = env.step(policy_function.compute_actions(o, policy_id="default"))
        ep_step += 1
        for kkk, ddd in d.items():
            if kkk != "__all__" and ddd:
                ep_success += 1 if info[kkk]["arrive_dest"] else 0
                ep_agent += 1
        print(d["__all__"])
        if d["__all__"]:  # This is important!
            print(
                {
                    "total agents": ep_agent,
                    "existing agents": len(o),
                    "success rate": ep_success / ep_agent if ep_agent > 0 else None,
                    "ep step": ep_step
                }
            )
            ep_success = 0
            ep_step = 0
            ep_agent = 0
            o,info = env.reset()
            d = {"__all__": False}
        if args.use_native_render:
            env.render(
                text={
                    "total agents": ep_agent,
                    "existing agents": len(o),
                    "success rate": ep_success / ep_agent if ep_agent > 0 else None,
                    "ep step": ep_step,
                    "Press": "Q to switch view"
                }
            )
        else:
            env.render(mode="top_down", num_stack=25)
    env.close()