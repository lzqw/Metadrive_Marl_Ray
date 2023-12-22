import copy
import os
import pickle

import numpy as np
from ray import tune
from ray.tune import CLIReporter

from marlray.utils.utils import initialize_ray

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def train(
    trainer,
    config,
    stop,
    exp_name,
    num_seeds=1,
    num_gpus=0,
    test_mode=False,
    suffix="",
    checkpoint_freq=2,
    keep_checkpoints_num=None,
    start_seed=0,
    local_mode=False,
    save_pkl=True,
    custom_callback=None,
    max_failures=1,
    **kwargs
):
    # initialize ray

    initialize_ray(test_mode=test_mode, local_mode=local_mode, num_gpus=num_gpus)

    # prepare config
    used_config = {
        "seed": tune.grid_search([i * 100 + start_seed for i in range(num_seeds)]) if num_seeds is not None else None,
        "log_level": "DEBUG" if test_mode else "INFO",
        "callbacks": custom_callback if custom_callback else False,  # Must Have!
    }
    if custom_callback is False:
        used_config.pop("callbacks")
    if config:
        used_config.update(config)
    config = copy.deepcopy(used_config)

    if isinstance(trainer, str):
        trainer_name = trainer
    elif hasattr(trainer, "_name"):
        trainer_name = trainer._name
    else:
        trainer_name = trainer.__name__

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if (keep_checkpoints_num is not None) and (not test_mode) and (keep_checkpoints_num != 0):
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns=metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    progress_reporter.add_metric_column("takeover")
    progress_reporter.add_metric_column("rc")
    kwargs["progress_reporter"] = progress_reporter

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir=root+'/ray_results',
        **kwargs
    )

    # save training progress as insurance
    if save_pkl:
        pkl_path = "{}-{}{}.pkl".format(exp_name, trainer_name, "" if not suffix else "-" + suffix)
        with open(pkl_path, "wb") as f:
            data = analysis.trial_dataframes
            pickle.dump(data, f)
            print("Result is saved at: <{}>".format(pkl_path))
    return analysis
