import os.path as osp
import numpy as np
root = osp.dirname(osp.abspath(osp.dirname(__file__)))

def get_policy_function(model_name: str, checkpoint_dir_name="checkpoints"):
    # checkpoint_name is like: {ALGO}_{ENV}_{INDEX}.npz
    global _checkpoints_buffers
    if model_name not in _checkpoints_buffers:
        path = osp.join(root, checkpoint_dir_name, model_name + ".npz")
        w = np.load(path)
        w = {k: w[k] for k in w.files}
        _checkpoints_buffers[model_name] = w
    else:
        w = _checkpoints_buffers[model_name]

    if model_name.startswith("ccppo"):
        return lambda obs: _compute_actions_for_torch_policy(w, obs)
    elif model_name.startswith("ippo"):
        return lambda obs: _compute_actions_for_tf_policy(w, obs, policy_name="default", layer_name_suffix="")
    elif model_name.startswith("cl"):
        return lambda obs: _compute_actions_for_tf_policy(w, obs, policy_name="default", layer_name_suffix="")
    elif model_name.startswith("copo"):
        return lambda obs: _compute_actions_for_tf_policy(w, obs, policy_name="default", layer_name_suffix="_1")
    else:
        raise ValueError("Unknown model: ", model_name)

def _compute_actions_for_torch_policy2(weights, obs, policy_name=None, layer_name_suffix=None, deterministic=None):
    obs = np.asarray(obs)
    assert obs.ndim == 2
    x = np.matmul(obs, weights["_hidden_layers.0._model.0.weight"].T) + weights["_hidden_layers.0._model.0.bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["_hidden_layers.1._model.0.weight"].T) + weights["_hidden_layers.1._model.0.bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["_logits._model.0.weight"].T) + weights["_logits._model.0.bias"]
    mean, log_std = np.split(x, 2, axis=1)
    if deterministic:
        return mean
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    return action