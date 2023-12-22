from typing import Type

import gymnasium as gym
import numpy as np
from marlray.algo.ippo import IPPOTrainer, IPPOConfig
from marlray.utils.callbacks import MultiAgentDrivingCallbacks
from marlray.utils.env_wrappers import get_rllib_compatible_env
from marlray.utils.train import train
from marlray.utils.utils import get_train_parser
from gymnasium.spaces import Box
from ray import tune
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()

CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"


class MAPPOConfig(IPPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or MAPPOConfig)
        self.counterfactual = True
        self.num_neighbours = 4
        self.fuse_mode = "mf"  # In ["concat", "mf", "none"]
        self.mf_nei_distance = 10
        self.old_value_loss = True
        self.update_from_dict({"model": {"custom_model": "cc_model"}})

    def validate(self):
        super().validate()
        assert self["fuse_mode"] in ["mf", "concat", "none"]
        self.model["custom_model_config"]["fuse_mode"] = self["fuse_mode"]
        self.model["custom_model_config"]["counterfactual"] = self["counterfactual"]
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]


def get_centralized_critic_obs_dim(
        observation_space_shape, action_space_shape, counterfactual, num_neighbours, fuse_mode
):
    """Get the centralized critic"""
    if fuse_mode == "concat":
        pass
    elif fuse_mode == "mf":
        num_neighbours = 1
    elif fuse_mode == "none":  # Do not use centralized critic
        num_neighbours = 0
    else:
        raise ValueError("Unknown fuse mode: ", fuse_mode)
    num_neighbours += 1
    centralized_critic_obs_dim = num_neighbours * observation_space_shape.shape[0]
    if counterfactual:  # Do not include ego action!
        centralized_critic_obs_dim += (num_neighbours - 1) * action_space_shape.shape[0]
    return centralized_critic_obs_dim


class CCModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(
            self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
            model_config: ModelConfigDict, name: str
    ):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # ========== Our Modification: We compute the centralized critic obs size here! ==========
        centralized_critic_obs_dim = self.get_centralized_critic_obs_dim()

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # ========== Our Modification ==========
            # Note: We use centralized critic obs size as the input size of critic!
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = centralized_critic_obs_dim
            assert prev_vf_layer_size > 0

            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        )

        self.view_requirements[CENTRALIZED_CRITIC_OBS] = ViewRequirement(
            space=Box(obs_space.low[0], obs_space.high[0], shape=(centralized_critic_obs_dim,))
        )

        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(space=action_space)

    def get_centralized_critic_obs_dim(self):
        return get_centralized_critic_obs_dim(
            self.obs_space, self.action_space, self.model_config["custom_model_config"]["counterfactual"],
            self.model_config["custom_model_config"]["num_neighbours"],
            self.model_config["custom_model_config"]["fuse_mode"]
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        features = self._hidden_layers(obs)
        logits = self._logits(features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        raise ValueError(
            "Centralized Value Function should not be called directly! "
            "Call central_value_function(cobs) instead!"
        )

    def central_value_function(self, obs):
        assert self._value_branch is not None
        return torch.reshape(self._value_branch(self._value_branch_separate(obs)), [-1])


ModelCatalog.register_custom_model("cc_model", CCModel)


def concat_mappo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Concat the neighbors' observations"""
    for index in range(sample_batch.count):
        environmental_time_step = sample_batch["t"][index]
        neighbours = sample_batch['infos'][index]["neighbours"]

        # Note that neighbours returned by the environment are already sorted based on their
        # distance to the ego vehicle whose info is being used here.
        for nei_count, nei_name in enumerate(neighbours):
            if nei_count >= policy.config["num_neighbours"]:
                break

            nei_act = None
            nei_obs = None
            if nei_name in other_agent_batches:
                _, nei_batch = other_agent_batches[nei_name]

                match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                if len(match_its_step) == 0:
                    pass
                elif len(match_its_step) > 1:
                    raise ValueError()
                else:
                    new_index = match_its_step[0]
                    nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                    nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

            if nei_obs is not None:
                start = odim + nei_count * other_info_dim
                sample_batch[CENTRALIZED_CRITIC_OBS][index, start:start + odim] = nei_obs
                if policy.config[COUNTERFACTUAL]:
                    sample_batch[CENTRALIZED_CRITIC_OBS][index, start + odim:start + odim + adim] = nei_act
                    assert start + odim + adim == start + other_info_dim
                else:
                    assert start + odim == start + other_info_dim
    return sample_batch

def mean_field_mappo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Average the neighbors' observations and probably actions."""
    # Note: Average other's observation might not be a good idea.
    # Maybe we can do some feature extraction before averaging all observations

    assert odim + other_info_dim == sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
    for index in range(sample_batch.count):

        environmental_time_step = sample_batch["t"][index]

        neighbours = sample_batch['infos'][index]["neighbours"]
        neighbours_distance = sample_batch['infos'][index]["neighbours_distance"]

        obs_list = []
        act_list = []

        for nei_count, (nei_name, nei_dist) in enumerate(zip(neighbours, neighbours_distance)):
            if nei_dist > policy.config["mf_nei_distance"]:
                continue

            nei_act = None
            nei_obs = None
            if nei_name in other_agent_batches:
                _, nei_batch = other_agent_batches[nei_name]

                match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                if len(match_its_step) == 0:
                    pass
                elif len(match_its_step) > 1:
                    raise ValueError()
                else:
                    new_index = match_its_step[0]
                    nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                    nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

            if nei_obs is not None:
                obs_list.append(nei_obs)
                act_list.append(nei_act)

        if len(obs_list) > 0:
            sample_batch[CENTRALIZED_CRITIC_OBS][index, odim:odim + odim] = np.mean(obs_list, axis=0)
            if policy.config[COUNTERFACTUAL]:
                sample_batch[CENTRALIZED_CRITIC_OBS][index, odim + odim:odim + odim + adim] = np.mean(act_list, axis=0)

    return sample_batch


def get_mappo_env(env_class):
    return get_rllib_compatible_env(get_maenv(env_class))