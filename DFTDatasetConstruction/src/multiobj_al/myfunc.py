#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
@edit: Ashley D.
"""
import logging
import pickle
import time

import numpy as np
import pandas as pd
from sklearn import metrics

logger = logging.getLogger(__name__)


def get_scores(
        model, X_train, y_train, X_test, y_test,
        X_val=None, y_val=None, alignn_model=None, dataset=None,
        sample_ids=None, sample_ids_val=None, train_target=None
        ):

    metrics_dict = {}

    start_time = time.time()
    model.fit(X_train, y_train)

    def _coerce_prediction_output(pred):
        """Extract primary prediction array from model outputs like (mean, std)."""
        if isinstance(pred, (tuple, list)):
            if len(pred) == 0:
                return np.asarray([])
            pred = pred[0]
        return np.asarray(pred)

    y_pred = _coerce_prediction_output(model.predict(X_test))
    if (X_val is None) or (len(X_val) == 0):
        y_pred_val = -1
    else:
        y_pred_val = _coerce_prediction_output(model.predict(X_val))

    y_test_arr = np.asarray(y_test)
    if len(y_pred) != len(y_test_arr):
        logger.warning(
            'Skipping test metrics due to prediction/target size mismatch: '
            f'len(y_pred)={len(y_pred)}, len(y_test)={len(y_test_arr)}'
        )
        maes = np.nan
        rmse = np.nan
        r2 = np.nan
    else:
        maes = metrics.mean_absolute_error(y_test_arr, y_pred)
        rmse = metrics.root_mean_squared_error(y_test_arr, y_pred)
        r2 = metrics.r2_score(y_test_arr, y_pred)

    metrics_dict['model_maes'] = maes
    metrics_dict['model_rmse'] = rmse
    metrics_dict['model_r2'] = r2

    logger.info(f'Test scores: MAE={maes:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}')

    if alignn_model is not None:
        if sample_ids is None:
            logger.warning(
                'Skipping ALIGNN fit: sample_ids mapping not available. '
                'ALIGNN metrics will be set to NaN.'
            )
            metrics_dict['alignn_maes'] = np.nan
            metrics_dict['alignn_rmse'] = np.nan
            metrics_dict['alignn_r2'] = np.nan
        else:
            logger.info(f'Fitting the ALIGNN model with {dataset} ...')

            try:
                with open(f'alldataRAW_{dataset}_ALIGNN.pkl', 'rb') as f:
                    d = pickle.load(f)
                df = pd.DataFrame(d)

                # Auto-detect the ID column rather than assuming 'reference'
                _id_candidates = ['reference', 'jid', 'id', 'material_id', 'task_id']
                raw_id_col = next((c for c in _id_candidates if c in df.columns), None)
                if raw_id_col is None:
                    logger.warning(
                        'No recognised ID column in ALIGNN pickle (columns: %s). '
                        'ALIGNN metrics will be set to NaN.', list(df.columns)
                    )
                    metrics_dict['alignn_maes'] = np.nan
                    metrics_dict['alignn_rmse'] = np.nan
                    metrics_dict['alignn_r2'] = np.nan
                else:
                    common_train_indices = X_train.index.intersection(sample_ids.index)
                    if len(common_train_indices) == 0:
                        logger.warning(
                            'Skipping ALIGNN fit: no matching indices between training data and sample_ids. '
                            'ALIGNN metrics will be set to NaN.'
                        )
                        metrics_dict['alignn_maes'] = np.nan
                        metrics_dict['alignn_rmse'] = np.nan
                        metrics_dict['alignn_r2'] = np.nan
                    else:
                        samples_for_ALIGNN = sample_ids.loc[common_train_indices].dropna().tolist()

                        val_samples_for_ALIGNN = []
                        common_val_indices = pd.Index([])
                        if X_val is not None and len(X_val) > 0:
                            common_val_indices = X_val.index.intersection(sample_ids.index)
                            if len(common_val_indices) > 0:
                                val_samples_for_ALIGNN = sample_ids.loc[common_val_indices].dropna().tolist()

                        samples_for_ALIGNN = [
                            item for ref in samples_for_ALIGNN
                            for item in (ref if isinstance(ref, (list, tuple)) else [ref])
                        ]
                        val_samples_for_ALIGNN = [
                            item for ref in val_samples_for_ALIGNN
                            for item in (ref if isinstance(ref, (list, tuple)) else [ref])
                        ]

                        df_train = df[df[raw_id_col].isin(samples_for_ALIGNN)]
                        df_val = df[df[raw_id_col].isin(val_samples_for_ALIGNN)] if X_val is not None else None

                        if df_train.empty:
                            logger.warning(
                                'Skipping ALIGNN fit: no ALIGNN-compatible training rows were found. '
                                'ALIGNN metrics will be set to NaN.'
                            )
                            metrics_dict['alignn_maes'] = np.nan
                            metrics_dict['alignn_rmse'] = np.nan
                            metrics_dict['alignn_r2'] = np.nan
                        else:
                            X_ALIGNN = df_train['atoms']
                            ref_to_y_train = {
                                r: y_train.loc[idx]
                                for idx, refs in sample_ids.loc[common_train_indices].dropna().items()
                                for r in (refs if isinstance(refs, (list, tuple)) else [refs])
                            }
                            y_ALIGNN = df_train[raw_id_col].map(ref_to_y_train)

                            # Drop rows whose reference ID had no match in ref_to_y_train.
                            # Such rows get NaN from .map() and will corrupt the loss function.
                            nan_mask = y_ALIGNN.isna()
                            if nan_mask.any():
                                logger.warning(
                                    "Dropping %d ALIGNN training row(s) with NaN target "
                                    "(reference ID not found in sample_ids mapping).",
                                    nan_mask.sum(),
                                )
                                X_ALIGNN = X_ALIGNN[~nan_mask]
                                y_ALIGNN = y_ALIGNN[~nan_mask]

                            x_ALIGNN_val = df_val['atoms'] if df_val is not None else None
                            if df_val is not None and len(common_val_indices) > 0 and y_val is not None:
                                ref_to_y_val = {
                                    r: y_val.loc[idx]
                                    for idx, refs in sample_ids.loc[common_val_indices].dropna().items()
                                    for r in (refs if isinstance(refs, (list, tuple)) else [refs])
                                }
                                y_ALIGNN_val = df_val[raw_id_col].map(ref_to_y_val)
                                nan_mask_val = y_ALIGNN_val.isna()
                                if nan_mask_val.any():
                                    logger.warning(
                                        "Dropping %d ALIGNN validation row(s) with NaN target.",
                                        nan_mask_val.sum(),
                                    )
                                    x_ALIGNN_val = x_ALIGNN_val[~nan_mask_val]
                                    y_ALIGNN_val = y_ALIGNN_val[~nan_mask_val]
                            else:
                                y_ALIGNN_val = None

                            if X_ALIGNN.empty:
                                raise ValueError(
                                    "All ALIGNN training rows had NaN targets after mapping; "
                                    "no samples left to train on."
                                )

                            logger.info(
                                "Training ALIGNN model on %d samples for target %s",
                                len(X_ALIGNN), train_target
                            )

                            alignn_model.fit(X_ALIGNN, y_ALIGNN)

                            logger.info("ALIGNN training completed.")

                            if x_ALIGNN_val is None or len(x_ALIGNN_val) == 0 or y_ALIGNN_val is None:
                                logger.warning(
                                    'Skipping ALIGNN validation metrics: no ALIGNN-compatible validation rows.'
                                )
                                metrics_dict['alignn_maes'] = np.nan
                                metrics_dict['alignn_rmse'] = np.nan
                                metrics_dict['alignn_r2'] = np.nan
                            else:
                                logger.info(
                                    "Predicting with ALIGNN model on %d validation samples...",
                                    len(x_ALIGNN_val)
                                )
                                alignn_y_pred = alignn_model.predict(x_ALIGNN_val)
                                alignn_maes = metrics.mean_absolute_error(y_ALIGNN_val, alignn_y_pred)
                                alignn_rmse = metrics.root_mean_squared_error(y_ALIGNN_val, alignn_y_pred)
                                alignn_r2 = metrics.r2_score(y_ALIGNN_val, alignn_y_pred)

                                logger.info(
                                    "ALIGNN validation metrics: MAE=%f, RMSE=%f, R2=%f",
                                    alignn_maes, alignn_rmse, alignn_r2
                                )
                                metrics_dict['alignn_maes'] = alignn_maes
                                metrics_dict['alignn_rmse'] = alignn_rmse
                                metrics_dict['alignn_r2'] = alignn_r2
            except FileNotFoundError:
                logger.warning(
                    f'Skipping ALIGNN fit: ALIGNN data file not found at alldataRAW_{dataset}_ALIGNN.pkl. '
                    'ALIGNN metrics will be set to NaN.'
                )
                metrics_dict['alignn_maes'] = np.nan
                metrics_dict['alignn_rmse'] = np.nan
                metrics_dict['alignn_r2'] = np.nan
            except Exception as e:
                logger.warning(
                    f'Skipping ALIGNN fit due to error: {e}. '
                    'ALIGNN metrics will be set to NaN.'
                )
                metrics_dict['alignn_maes'] = np.nan
                metrics_dict['alignn_rmse'] = np.nan
                metrics_dict['alignn_r2'] = np.nan

    if (X_val is not None) and (len(X_val) > 0) and (y_val is not None):
        y_pred = _coerce_prediction_output(model.predict(X_val))
        y_val_arr = np.asarray(y_val)
        if len(y_pred) == len(y_val_arr):
            maes_val = metrics.mean_absolute_error(y_val_arr, y_pred)
            rmse_val = metrics.root_mean_squared_error(y_val_arr, y_pred)
            r2_val = metrics.r2_score(y_val_arr, y_pred)
            logger.info(f'Val scores: MAE={maes_val:.3f}, RMSE={rmse_val:.3f}, R2={r2_val:.3f}')
        else:
            logger.warning(
                'Skipping validation metrics due to prediction/target size mismatch: '
                f'len(y_pred)={len(y_pred)}, len(y_val)={len(y_val_arr)}'
            )

    logger.info("--- %.2f seconds ---", time.time() - start_time)
    logger.info('')

    metrics_dict['model_pred'] = y_pred_val

    return metrics_dict


def screen_large_structures(df, sample_ids, alignn_pickle_path, max_atoms, id_column='reference'):
    """Remove structures with more than max_atoms atoms from the dataset.

    Structures that are too large create oversized graph batches in ALIGNN
    training and are dropped from the pool entirely before the AL loop starts.

    Parameters
    ----------
    df : pd.DataFrame
        Main feature/target DataFrame, indexed consistently with sample_ids.
    sample_ids : pd.Series
        Maps DataFrame index → ALIGNN reference ID(s). Each value may be a
        single reference string or a list/tuple of strings.
    alignn_pickle_path : str
        Path to the ALIGNN pickle file (columns: id_column, 'atoms', ...).
    max_atoms : int
        Structures with strictly more than this many atoms are removed.
    id_column : str
        Name of the reference-ID column in the ALIGNN pickle.

    Returns
    -------
    df_filtered : pd.DataFrame
    sample_ids_filtered : pd.Series
    """
    try:
        with open(alignn_pickle_path, 'rb') as f:
            alignn_data = pickle.load(f)
        alignn_df = pd.DataFrame(alignn_data)
    except FileNotFoundError:
        logger.warning(
            f'ALIGNN pickle not found at {alignn_pickle_path}; '
            'skipping large-structure screening.'
        )
        return df, sample_ids

    def _safe_num_atoms(atoms_obj):
        try:
            return atoms_obj.num_atoms
        except AttributeError:
            try:
                return len(atoms_obj)
            except Exception:
                return None

    alignn_df['_n_atoms'] = alignn_df['atoms'].apply(_safe_num_atoms)

    # Resolve the ID column, falling back to common alternatives if the configured name is absent
    if id_column not in alignn_df.columns:
        candidates = ['jid', 'reference', 'id', 'material_id', 'task_id']
        found = next((c for c in candidates if c in alignn_df.columns), None)
        if found is None:
            logger.warning(
                "ID column '%s' not found in ALIGNN pickle (columns: %s); "
                'skipping large-structure screening.',
                id_column, list(alignn_df.columns),
            )
            return df, sample_ids
        logger.warning(
            "ID column '%s' not found in ALIGNN pickle; using '%s' instead.",
            id_column, found,
        )
        id_column = found

    large_refs = set(
        alignn_df.loc[alignn_df['_n_atoms'] > max_atoms, id_column].dropna()
    )

    if not large_refs:
        logger.info(f'No structures with more than {max_atoms} atoms found.')
        return df, sample_ids

    def _refs(val):
        return val if isinstance(val, (list, tuple)) else [val]

    large_indices = [
        idx for idx, refs in sample_ids.items()
        if any(r in large_refs for r in _refs(refs))
    ]

    if large_indices:
        logger.warning(
            'Screening %d structures with >%d atoms from dataset. Sample IDs: %s',
            len(large_indices), max_atoms, large_indices,
        )
        df = df.drop(index=large_indices)
        sample_ids = sample_ids.drop(index=large_indices)

    logger.info('After large-structure screening: %d samples remain.', len(df))
    return df, sample_ids


def drop_failed_structures(df, label_stuc=None):
    if label_stuc is None:
        label_stuc = df.columns[-273:-145]
    df_failed_struct = df[df[label_stuc].isnull().any(axis=1)]
    num_tot = df.shape[0]
    logger.info(f'Total: {num_tot}')
    num_failed = df_failed_struct.shape[0]
    logger.warning(f'Failed structural featurization: {num_failed}')
    logger.info(f'Sucess: {num_tot - num_failed}')
    return df.drop(df_failed_struct.index)
