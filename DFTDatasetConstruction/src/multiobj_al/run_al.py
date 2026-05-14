#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu January 26 15:48:51 2023
@author: kangming
@edit: daniel
@edit: Ashley D.
@edit: Hongchen W.
"""

import os

# Must be set before any CUDA call to enable deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Reduce GPU memory fragmentation from repeated alloc/free cycles during AL.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# pylint: disable=wrong-import-position
import argparse
import logging
import pickle
import time
import traceback

from jarvis.db.figshare import data as jarvis_figshare_data

from .alignn_wrapper import AlignnLayerNorm, NfflrAlignnConfig
from .datasets import load_config
from .distill import get_data, return_model
from .policies.diversity_policy import DiversityPolicy
from .policies.multiobj_policy import MultiObjPolicy
from .policies.qbc_policy import QBCPolicy
from .policies.qvendi_policy import QVendiPolicy
from .policies.random_policy import RandomPolicy
# pylint: enable=wrong-import-position

# Configure initial logging (console only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info('Dependencies imported successfully!')

def build_parser():
    parser = argparse.ArgumentParser(
        description='Grow datasets with different growing methods.'
    )

    parser.add_argument('--growingCriteria', type=str, required=False, default='grow_MultiObj_All',
                        help=(
                            "Specify the growing criteria: random, QBC, grow_MultiObj_All, "
                            "grow_MultiObj2Outcomes, grow_QVendi, grow_QVendi2Outcomes, "
                            "grow_EA_QBC_1Obj_withDiversity_from3Outcomes, "
                            "grow_EA_QBC_2Obj_twoTargets_from3Outcomes, "
                            "grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes"
                        ))
    parser.add_argument('--config', type=str, required=True,
                        help="Path to dataset configuration JSON file")
    parser.add_argument('--target', type=str, required=False, default='e_form',
                        help=(
                            "Specify the target property to optimize: e_form, bulk_modulus, "
                            "bandgap, eform_bulk_modulus, bandgap_eform, bandgap_bulkmodulus"
                        ))
    parser.add_argument('--outputDir', type=str, required=False, default='output',
                        help="Specify the output directory to save results")
    parser.add_argument('--stepFrac', type=float, nargs='+', required=False, default=[0.01],
                        help=(
                            "Total fraction of training data to consume at each AL checkpoint. "
                            "A single value (e.g., 0.1) is used as a fixed step size repeated "
                            "until the pool is exhausted. Multiple strictly-increasing values "
                            "(e.g., 0.1 0.3 0.5 1.0) define exact milestones: the model is "
                            "evaluated at each cumulative percentage and the run stops after "
                            "the last milestone."
                        ))
    parser.add_argument('--randomSeed', type=int, required=False, default=0,
                        help="Specify the random seed for reproducibility")
    parser.add_argument('--alignn', required=False, default=False,
                        type=lambda v: v.lower() in ('true', '1', 'yes'),
                        help="Specify whether to use the ALIGNN model (True/False)")
    parser.add_argument('--early-stopping-patience', type=int, required=False, default=None,
                        help="Early stopping patience (epochs with no val improvement). 0 disables. "
                             "Overrides config.json only if explicitly passed.")
    parser.add_argument('--early-stopping-min-delta', type=float, required=False, default=None,
                        help="Minimum val improvement to reset early-stopping counter. "
                             "Overrides config.json only if explicitly passed.")
    parser.add_argument('--max-atoms', type=int, required=False, default=None,
                        help="Remove structures with more than this many atoms before the AL loop. "
                             "Uses the 'nsites' column when available, otherwise falls back to the "
                             "ALIGNN pickle. Overrides the value in the dataset config.")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    growingCriteria = args.growingCriteria
    config_path = args.config
    target = args.target
    outputDir = args.outputDir
    # args.stepFrac contains cumulative data fractions (milestones), e.g. [0.1, 0.3, 0.5, 1.0].
    # Convert to additive step fracs consumed by the grower.
    _milestones = args.stepFrac
    if len(_milestones) == 1:
        fltStepFrac = _milestones          # single value: step size repeated indefinitely
        n_al_steps = None                  # no limit on number of selections
    else:
        for i in range(1, len(_milestones)):
            if _milestones[i] <= _milestones[i - 1]:
                raise ValueError(
                    f"--stepFrac milestones must be strictly increasing, got {_milestones}"
                )
        fltStepFrac = [_milestones[0]] + [
            round(_milestones[i] - _milestones[i - 1], 10)
            for i in range(1, len(_milestones))
        ]
        n_al_steps = len(_milestones) - 1  # evaluate at exactly this many checkpoints after init
    intRandomSeed = args.randomSeed
    use_alignn = args.alignn

    # Load dataset configuration
    logger.info(f'Loading dataset configuration from {config_path}')
    dataset_config = load_config(config_path)
    dataset = dataset_config.name
    logger.info(f'Dataset: {dataset}')
    logger.info(f'Targets: {dataset_config.target_columns}')

    if args.max_atoms is not None:
        dataset_config.alignn.max_atoms = args.max_atoms
        logger.info(f'Large-structure screening enabled: max_atoms={args.max_atoms}')

    SINGLE_TARGETS = set(dataset_config.target_columns)
    PAIR_TARGETS = {
        f'{t1}_{t2}' for i, t1 in enumerate(dataset_config.target_columns)
        for t2 in dataset_config.target_columns[i+1:]
    }

    allowed_targets_by_criteria = {
        'QBC': SINGLE_TARGETS,
        'grow_QVendi': SINGLE_TARGETS,
        'grow_EA_QBC_1Obj_withDiversity_from3Outcomes': SINGLE_TARGETS,
        'grow_MultiObj2Outcomes': PAIR_TARGETS,
        'grow_QVendi2Outcomes': PAIR_TARGETS,
        'grow_EA_QBC_2Obj_twoTargets_from3Outcomes': PAIR_TARGETS,
        'grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes': PAIR_TARGETS,
        # These criteria do not consume target directly, but keep target constrained for consistency.
        'random': SINGLE_TARGETS,
        'grow_MultiObj_All': SINGLE_TARGETS,
    }

    if growingCriteria in allowed_targets_by_criteria:
        allowed_targets = allowed_targets_by_criteria[growingCriteria]
        if target not in allowed_targets:
            raise ValueError(
                f"Invalid target '{target}' for growingCriteria '{growingCriteria}'. "
                f"Allowed targets: {sorted(allowed_targets)}"
            )

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Add file handler to save logs to output directory
    log_filename = os.path.join(outputDir, f'run_al_{dataset}_{growingCriteria}_{intRandomSeed}.log')
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f'Log file created: {log_filename}')

    random_state = 0

    model = {}
    save_raw_data = False

    for modelname in ['xgb', 'rf', 'alignn']:
        if modelname == 'alignn':
            if use_alignn:
                _config_path = os.path.join(
                    os.path.dirname(__file__), 'configs', 'config.json'
                )
                cfg = (
                    NfflrAlignnConfig.from_json(_config_path)
                    if os.path.exists(_config_path)
                    else NfflrAlignnConfig()
                )
                cfg.output_dir = os.path.join(outputDir, "output_alignn")
                # Apply early-stopping overrides only when explicitly passed on the CLI
                if args.early_stopping_patience is not None:
                    cfg.early_stopping_patience = args.early_stopping_patience
                if args.early_stopping_min_delta is not None:
                    cfg.early_stopping_min_delta = args.early_stopping_min_delta
                model[modelname] = AlignnLayerNorm(cfg)
                save_raw_data = True

                alignn_pkl = f'alldataRAW_{dataset}_ALIGNN.pkl'
                if alignn_pkl not in os.listdir('.'):
                    if dataset_config.alignn.jarvis_api_name:
                        jarvis_api_name = dataset_config.alignn.jarvis_api_name
                        logger.info(
                            "ALIGNN data file %s not found. Downloading %s from JARVIS figshare defined in config...",
                            alignn_pkl, jarvis_api_name
                        )
                        d = jarvis_figshare_data(jarvis_api_name)
                        with open(alignn_pkl, 'wb') as f:
                            pickle.dump(d, f)
                        logger.info(f"Raw ALIGNN data saved to {alignn_pkl}")
                    else:
                        logger.warning(
                            "ALIGNN enabled but no jarvis_api_name in config. Cannot download ALIGNN data.",
                            "If using a custom dataset, please generate the ALIGNN features and save them as a \
                            pickle file named %s in the current directory.",
                            alignn_pkl  
                        )
                        continue
            else:
                logger.info("Skipping ALIGNN model (--alignn not set).")
                continue
        else:
            model[modelname] = return_model(modelname, random_state)

    logger.info('Models imported successfully!')
    logger.info(f"GrowingCriteria is {growingCriteria}")
    logger.info(f"Random seed: {intRandomSeed}")

    # Training two outcomes training cases
    target1 = None
    target2 = None
    if growingCriteria in ('grow_MultiObj2Outcomes', 'grow_QVendi2Outcomes',
                        'grow_EA_QBC_2Obj_twoTargets_from3Outcomes',
                        'grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes'):
        # Parse compound targets like 'e_form_bulk_modulus' -> ('e_form', 'bulk_modulus')
        parts = target.split('_')
        if len(parts) >= 2:
            # Try to find valid target combinations from the dataset config
            # This is a heuristic: try different split points
            for split_idx in range(1, len(parts)):
                candidate1 = '_'.join(parts[:split_idx])
                candidate2 = '_'.join(parts[split_idx:])
                in_targets = (candidate1 in dataset_config.target_columns
                              and candidate2 in dataset_config.target_columns)
                if in_targets:
                    target1 = candidate1
                    target2 = candidate2
                    break

            if target1 is None or target2 is None:
                raise ValueError(
                    f"Could not parse compound target '{target}' into two valid targets "
                    f"from {dataset_config.target_columns}"
                )
        else:
            raise ValueError(
                f"Multi-objective criteria requires compound target "
                f"(e.g., 'e_form_bulk_modulus'), got '{target}'"
            )

    # Flexible target handling
    # The dataset config specifies which targets are available.
    # Targets are now stored in dictionaries for flexible handling.

    if model.get('alignn') is not None:
        logger.info("ALIGNN model found, loading data with alignn features.")
        make_alignn_features = True
    else:
        make_alignn_features = False

    data_splits = get_data(
        config=dataset_config, random_state=0, save_raw=save_raw_data, alignn_features=make_alignn_features,
    )

    alldata = data_splits['df']
    sample_ids_pool = data_splits['sample_ids']
    xTrainVal = data_splits['X_pool']
    xTest = data_splits['X_test']

    # Extract targets from dictionaries
    y_targets_train_val = data_splits['y_targets_pool']
    y_targets_test = data_splits['y_targets_test']

    # Handle 'compound possible' feature if it exists
    if 'compound possible' in xTrainVal.columns:
        xTrainVal['compound possible'] = xTrainVal['compound possible'].astype(float)
    if 'compound possible' in xTest.columns:
        xTest['compound possible'] = xTest['compound possible'].astype(float)

    logger.info(f"Saving all data to {outputDir}...")
    alldata.to_csv(os.path.join(outputDir, "alldata_" + dataset + ".csv"), index=False)
    xTrainVal.to_csv(os.path.join(outputDir, "xTrainVal_" + dataset + ".csv"), index=False)
    xTest.to_csv(os.path.join(outputDir, "xTest_" + dataset + ".csv"))

    # Save targets
    for target_name, target_series in y_targets_train_val.items():
        target_series.to_csv(os.path.join(outputDir, f"{target_name}TrainVal_{dataset}.csv"))
    for target_name, target_series in y_targets_test.items():
        target_series.to_csv(os.path.join(outputDir, f"{target_name}Test_{dataset}.csv"))

    logger.info("Finished Saving all data")

    logger.info(f'Dataset: {dataset} | Target: {target}')
    # print the size of the training sets
    logger.info(f'xTrainVal Length: {len(xTrainVal)}')
    tStart = time.time()
    try:
        common_kwargs = {
            'model1': model['rf'],
            'model2': model['xgb'],
            'x_train_val': xTrainVal,
            'y_targets_train_val': y_targets_train_val,
            'x_test': xTest,
            'y_targets_test': y_targets_test,
            'str_save_dir': outputDir,
            'str_model1_name': 'rf',
            'str_model2_name': 'xgb',
            'flt_step_frac': fltStepFrac,
            'n_al_steps': n_al_steps,
            'int_random_seed': intRandomSeed,
            'alignn_model': model.get('alignn', None),
            'sample_ids': sample_ids_pool,
            'dataset': dataset,
        }

        if growingCriteria == 'random':
            policy = RandomPolicy(
                str_save_name=f'{dataset}_{target}_random{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'QBC':
            policy = QBCPolicy(
                target=target,
                str_save_name=f'{dataset}_{target}_QBC{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_MultiObj_All':
            policy = MultiObjPolicy(
                str_save_name=f'{dataset}_{target}_rfMaxUncertainty{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_MultiObj2Outcomes':
            policy = MultiObjPolicy(
                target1=target1,
                target2=target2,
                str_save_name=f'{dataset}_{target}MultiObjective_OptDiffModelsOnly2Var{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_QVendi':
            policy = QVendiPolicy(
                target=target,
                str_save_name=f'{dataset}_{target}_QVendi{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_QVendi2Outcomes':
            policy = QVendiPolicy(
                target1=target1,
                target2=target2,
                str_save_name=f'{dataset}_{target}_QVendi2Outcomes{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_EA_QBC_1Obj_withDiversity_from3Outcomes':
            policy = DiversityPolicy(
                target=target,
                str_save_name=f'{dataset}_{target}_EA_1obj_div{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_EA_QBC_2Obj_twoTargets_from3Outcomes':
            policy = DiversityPolicy(
                target1=target1,
                target2=target2,
                use_diversity=False,
                str_save_name=f'{dataset}_{target}_EA_2obj_qbc{intRandomSeed}',
                **common_kwargs
            )
        elif growingCriteria == 'grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes':
            policy = DiversityPolicy(
                target1=target1,
                target2=target2,
                use_diversity=True,
                str_save_name=f'{dataset}_{target}_EA_2obj_div{intRandomSeed}',
                **common_kwargs
            )
        else:
            logger.error('Invalid growing criteria!')
            logger.error(
                'Valid criteria: random, QBC, grow_MultiObj_All, grow_MultiObj2Outcomes, '
                'grow_QVendi, grow_QVendi2Outcomes, '
                'grow_EA_QBC_1Obj_withDiversity_from3Outcomes, '
                'grow_EA_QBC_2Obj_twoTargets_from3Outcomes, '
                'grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes'
            )
            return 1

        # The policy is called here:
        policy.grow()

    except Exception as e:
        logger.error(f'Error during dataset growing: {e}')
        logger.error(traceback.format_exc())
        return 1

    tEnd = time.time()
    logger.info(f'Time elapsed: {round(tEnd - tStart, 2)} seconds')
    logger.info(f'Done growing by {growingCriteria} on {dataset}!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
