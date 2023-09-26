import itertools
import operator
import os
import shutil
import time
from functools import reduce
from typing import List, Union

import configargparse
import yaml
from flatten_dict import flatten, unflatten
from loguru import logger
from yacs.config import CfgNode as CN

from utils.cluster import execute_task_on_cluster
from utils.default_hparams import hparams


def parse_args():
    def add_common_cmdline_args(parser):
        # for cluster runs
        parser.add_argument('--cfg', required=True, type=str, help='cfg file path')
        parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
        parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
        parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
        parser.add_argument('--bid', type=int, default=10, help='amount of bid for cluster')
        parser.add_argument('--memory', type=int, default=64000, help='memory amount for cluster')
        parser.add_argument('--gpu_min_mem', type=int, default=12000, help='minimum amount of GPU memory')
        parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                            nargs='*', help='additional options to update config')
        parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')
        return parser

    # For Blender main parser
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of DECO'

    parser = configargparse.ArgumentParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='deco')

    parser = add_common_cmdline_args(parser)

    args = parser.parse_args()
    print(args, end='\n\n')

    return args

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()

def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()

def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()

def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []

    for k,v in flattened_config_dict.items():
        if isinstance(v,list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool) :
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v,list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ['True', 'False']):
                exp[param_name] = [True if x == 'True' else False for x in param_value]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False


        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params

def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)

def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)

def run_grid_search_experiments(
        args,
        script='train.py',
        change_wt_name=True
):
    cfg = yaml.safe_load(open(args.cfg))
    # parse config file to split into a list of configs with tuning hyperparameters separated
    # Also return the names of tuned hyperparameters hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=['TRAINING/DATASETS', 'TRAINING/DATASET_MIX_PDF', 'VALIDATION/DATASETS'],
    )
    logger.info(f'Grid search hparams: \n {hyperparams}')

    # The config file may be missing some default values, so we need to add them
    different_configs = [update_hparams_from_dict(c) for c in different_configs]
    logger.info(f'======> Number of experiment configurations is {len(different_configs)}')

    config_to_run = CN(different_configs[args.cfg_id])

    if args.cluster:
        execute_task_on_cluster(
            script=script,
            exp_name=config_to_run.EXP_NAME,
            output_dir=config_to_run.OUTPUT_DIR,
            condor_dir=config_to_run.CONDOR_DIR,
            cfg_file=args.cfg,
            num_exp=len(different_configs),
            bid_amount=args.bid,
            num_workers=config_to_run.DATASET.NUM_WORKERS,
            memory=args.memory,
            exp_opts=args.opts,
            gpu_min_mem=args.gpu_min_mem,
            gpu_arch=args.gpu_arch,
        )
        exit()

    # ==== create logdir using hyperparam settings
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{config_to_run.EXP_NAME}'
    wt_file = config_to_run.EXP_NAME + '_'
    for hp in hyperparams:
        v = get_from_dict(different_configs[args.cfg_id], hp.split('/'))
        logdir += f'_{hp.replace("/", ".").replace("_", "").lower()}-{v}'
        wt_file += f'{hp.replace("/", ".").replace("_", "").lower()}-{v}_'
    logdir = os.path.join(config_to_run.OUTPUT_DIR, logdir)
    os.makedirs(logdir, exist_ok=True)
    config_to_run.LOGDIR = logdir

    wt_file += 'best.pth'
    wt_path = os.path.join(os.path.dirname(config_to_run.TRAINING.BEST_MODEL_PATH), wt_file)
    if change_wt_name: config_to_run.TRAINING.BEST_MODEL_PATH = wt_path

    shutil.copy(src=args.cfg, dst=os.path.join(logdir, 'config.yaml'))

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)),
        os.path.join(config_to_run.LOGDIR, 'config_to_run.yaml')
    )

    return config_to_run