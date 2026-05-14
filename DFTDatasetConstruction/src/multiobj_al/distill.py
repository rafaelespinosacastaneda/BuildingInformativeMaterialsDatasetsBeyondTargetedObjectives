#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""

import json
import logging
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

from .datasets import DatasetConfig, get_loader
from .myfunc import drop_failed_structures, screen_large_structures

# Configure logging for this module
logger = logging.getLogger(__name__)



#%%

def return_model(modelname, random_state, config=None):
    if modelname == 'xgb':
        return xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.25,
        reg_lambda=0.01,reg_alpha=0.1,
        subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=4 ,device="cpu",
        random_state=random_state
        )

    if modelname == 'xgb_1tree':
        return xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.25,
        reg_lambda=0.01,
        num_parallel_tree=1, device="cpu",
        random_state=random_state
        )

    if modelname == 'rf':
        return RandomForestRegressor(
        n_estimators=100, max_features=1/3, n_jobs=-1, random_state=random_state
        )

    if modelname == 'alignn':
        if config is None:
            raise ValueError('Please provide config for ALIGNN model')
        try:
            from .alignn_wrapper import AlignnLayerNorm, NfflrAlignnConfig
            if isinstance(config, dict):
                config_obj = NfflrAlignnConfig(**{
                    k: v for k, v in config.items()
                    if k in NfflrAlignnConfig.__dataclass_fields__
                })
            elif isinstance(config, NfflrAlignnConfig):
                config_obj = config
            else:
                raise TypeError(
                    f"config must be a dict or NfflrAlignnConfig, got {type(config)}"
                )
            return AlignnLayerNorm(config_obj)
        except Exception as e:
            logger.error(f"Error creating ALIGNN model: {e}")
            return None

    raise ValueError(f'Unknown model: {modelname}')


#%%

'''
Define the function that returns the X and y
'''

def _getElementsFromFormula(strFormula: str):
    '''
    get the elements from a formula
    arguments
    ---------------
    strFormula : str
        the chemical formula
    returns
    ---------------
    lstElements : list
        the elements in the formula
    '''
    from  pymatgen.core.composition import Composition

    # get the composition
    comp = Composition(strFormula)
    # get the elements
    lstElements = list(comp.as_dict().keys())
    return lstElements

def getFeaturizedDataFrame(config: DatasetConfig,
                           save_raw=False):

    '''
    Get the featurized dataset using dataset configuration.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration object specifying source, features, targets, etc.
    save_raw : bool, optional
        Whether to save the raw data to CSV. Default is False.

    Returns
    -------
    lstFeatureColumns : list
        List of feature column names.
    dfX : pd.DataFrame
        The features and targets dataframe.
        Shape: (n, m) where n is number of samples, m is number of features + targets.
    sample_ids : pd.Series
        The sample IDs from the dataset.
    '''

    # Load data using appropriate loader
    loader = get_loader(config)
    df_raw = loader.load()

    # log the shape of the dataframe
    logging.debug(f'dataframe imported successfully - shape: {df_raw.shape}')
    if save_raw:
        logger.debug("SAVING RAW DATA")
        df_raw.to_csv("alldataRAW_"+config.name+".csv", index=True)
        logger.debug("DONE SAVING RAW DATA")

    # Apply column renames from config
    if config.preprocessing.column_renames:
        df_raw = df_raw.rename(columns=config.preprocessing.column_renames)
        logging.debug(f"Applied column renames: {config.preprocessing.column_renames}")

    # Get feature columns based on config
    if config.feature_columns.type == 'last_n':
        lstFeatureColumns = list(df_raw.columns[-config.feature_columns.n:])
    elif config.feature_columns.type == 'explicit':
        lstFeatureColumns = config.feature_columns.columns
    elif config.feature_columns.type == 'range':
        lstFeatureColumns = list(df_raw.columns[config.feature_columns.start:config.feature_columns.end])
    else:
        raise ValueError(f"Unsupported feature_columns type: {config.feature_columns.type}")

    # Check if ID column exists for sample tracking
    has_id_column = config.id_column in df_raw.columns

    # count the number of nan values in the formula column
    if config.formula_column in df_raw.columns:
        intNanCount_formula = df_raw[config.formula_column].isnull().sum()
        # log how many entries were dropped
        logging.info(f'{intNanCount_formula} entries dropped due to nan values in the formula column')
        # drop entries with nan values in the formula column
        df_temp = df_raw.dropna(subset=[config.formula_column])
        df_temp = df_temp.copy()
        # add the elements column to the temporary dataframe
        df_temp.loc[:,'elements'] = df_temp.loc[:,config.formula_column].apply(_getElementsFromFormula)
    else:
        logging.info(f"No formula column '{config.formula_column}' found in dataset")
        df_temp = df_raw.copy()

    # make smaller dataframe with only the columns of interest for ml
    ml_columns = config.target_columns + lstFeatureColumns

    df_temp_ml = df_temp[ml_columns]
    # log the shape of the dataframe
    logging.info(f'dataframe with only ML important columns created successfully - shape: {df_temp_ml.shape}')
    # count the number of nan values in the ML important columns
    intNanCount_ml = df_temp_ml.isnull().sum().sum()
    # log how many entries were dropped
    logging.info(f'{intNanCount_ml} entries dropped due to nan values in the ML important columns')
    # drop rows with nan values
    df_temp_ml = df_temp_ml.dropna()

    # Extract sample_ids AFTER all preprocessing, aligned to the post-dropna index
    if has_id_column:
        sample_ids = df_temp[config.id_column].loc[df_temp_ml.index]
        logging.debug(f'Sample IDs Found: {sample_ids[:5]}')
    else:
        sample_ids = None
        logging.info(f'No Sample IDs Found in dataset {config.name}. Proceeding without sample tracking.')

    # Return features and targets dataframe
    dfX = df_temp_ml

    # log the shape of the dataframe
    logging.info(f'features + targets: {dfX.shape}')
    return lstFeatureColumns, dfX, sample_ids



def get_data(
        config: DatasetConfig,
        test_size=0.1,
        random_state=0,
        fixed_train_ids=None,
        target2transfer=None,
        get_pruned_set=False,
        pruning_model=None,
        pruned_set_min_frac=0.3,
        col_X = None,
        standardized=True,
        data_dir = 'data',
        predefined_train_val_test = False,
        save_raw=False,
        alignn_features=False
        ):
    '''
    Load and split dataset based on configuration.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration object.
    test_size : float, optional
        Fraction of data for test set.  Default is 0.1.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    ... (other parameters remain the same)

    Returns
    -------
    dict
        Dictionary containing:
        - 'df': Full dataframe
        - 'X': Features
        - 'y_targets': Dict of {target_name: Series} for all targets
        - 'sample_ids': Sample IDs
        - 'X_pool', 'X_test': Train/test feature splits
        - 'y_targets_pool', 'y_targets_test': Train/test target dicts
    '''

    # Load data based on feature type
    if col_X is not None:
        raise NotImplementedError(f"col_X={col_X!r} is not supported. Use col_X=None.")
    lstFeatureColumns, df, sample_ids = getFeaturizedDataFrame(config=config, save_raw=save_raw)

    # Clean data: remove infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop NaNs in targets if requested
    if config.preprocessing.drop_na_targets:
        if target2transfer is None:
            df = df.dropna(subset=config.target_columns)
        else:
            df = df[~df[target2transfer].isna()]

    # Remove failed structures if configured
    if config.preprocessing.drop_failed_structures:
        if config.preprocessing.structural_feature_range is not None:
            # Use configured structural feature range
            start_idx, end_idx = config.preprocessing.structural_feature_range
            structural_features = list(df.columns[start_idx:end_idx])
            df = drop_failed_structures(df, label_stuc=structural_features)
        else:
            df = drop_failed_structures(df)

    # Apply quality filters from config
    for filter_config in config.preprocessing.filters:
        col = filter_config.column
        if col in df.columns:
            if filter_config.min is not None:
                df = df[df[col] >= filter_config.min]
                logging.info(f"Applied filter: {col} >= {filter_config.min}")
            if filter_config.max is not None:
                df = df[df[col] <= filter_config.max]
                logging.info(f"Applied filter: {col} <= {filter_config.max}")

    # Screen structures that are too large (for ALIGNN graph batching or general quality control)
    if config.alignn.max_atoms is not None:
        max_atoms = config.alignn.max_atoms
        aligned_sample_ids = sample_ids.loc[df.index] if sample_ids is not None else None

        if 'nsites' in df.columns:
            # Fast path: atom count is directly available in the main DataFrame
            large_mask = df['nsites'] > max_atoms
            n_large = int(large_mask.sum())
            if n_large > 0:
                large_indices = df.index[large_mask].tolist()
                logging.warning(
                    'Screening %d structures with >%d atoms (nsites column). Indices: %s',
                    n_large, max_atoms, large_indices,
                )
                df = df[~large_mask]
                if aligned_sample_ids is not None:
                    sample_ids = aligned_sample_ids[~large_mask]
            logging.info('After large-structure screening: %d samples remain.', len(df))
        elif aligned_sample_ids is not None and config.alignn.pickle_pattern is not None:
            # Fallback: derive atom counts from the ALIGNN pickle
            df, sample_ids = screen_large_structures(
                df,
                aligned_sample_ids,
                alignn_pickle_path=config.alignn.pickle_pattern,
                max_atoms=max_atoms,
                id_column=config.alignn.id_column_in_pickle,
            )
        else:
            logging.warning(
                'max_atoms=%d set but no nsites column or ALIGNN pickle available; '
                'skipping large-structure screening.', max_atoms
            )

    # Extract features and targets
    X = df[lstFeatureColumns]

    # Create dictionary of targets instead of y1, y2, y3
    y_targets = {target: df[target] for target in config.target_columns}

    # Filter sample_ids to match the cleaned dataframe indices if it exists
    if sample_ids is not None:
        sample_ids = sample_ids.loc[df.index]
    else:
        sample_ids = pd.Series(dtype='object', index=df.index)

    # Initialize result dictionary with base data
    result = {
        'df': df,
        'X': X,
        'y_targets': y_targets,
        'sample_ids': sample_ids,
        'config': config  # Include config for downstream use
    }

    # Check for data leakage
    if any(target in lstFeatureColumns for target in config.target_columns):
        warnings.warn("DATA LEAKAGE ALERT!")

    # Apply pruning if requested (NOT USED AT THE MOMENT)
    if get_pruned_set:
        raise NotImplementedError("Pruning functionality needs to be updated for flexible targets")

    # Helper function to add splits to result dictionary
    def _add_splits_to_result(
            X_pool, X_test, y_targets_pool, y_targets_test,
            X_val=None, y_targets_val=None, sample_ids=None, sample_ids_val=None
        ):

        result.update({
            'X_pool': X_pool,
            'X_test': X_test,
            'y_targets_pool': y_targets_pool,
            'y_targets_test': y_targets_test,
            'sample_ids': sample_ids
        })

        if X_val is not None:
            result.update({
                'X_val': X_val,
                'y_targets_val': y_targets_val,
                'sample_ids_val': sample_ids_val
            })

    # Perform train-val-test split based on configuration
    if fixed_train_ids is None:
        if predefined_train_val_test:
            # Load predefined splits from JSON
            with open(f'{data_dir}/{config.name}/train_val_test.json', 'r', encoding='utf-8') as f:
                train_val_test = json.load(f)
                train_ids = list(train_val_test["train"].keys())
                val_ids = list(train_val_test["val"].keys())
                test_ids = list(train_val_test["test"].keys())

            # Extract data for each split
            X_pool, X_val, X_test = X.loc[train_ids], X.loc[val_ids], X.loc[test_ids]

            # Create target dictionaries for each split
            y_targets_pool = {target: y_targets[target].loc[train_ids] for target in config.target_columns}
            y_targets_val = {target: y_targets[target].loc[val_ids] for target in config.target_columns}
            y_targets_test = {target: y_targets[target].loc[test_ids] for target in config.target_columns}

            sample_ids_val = sample_ids.loc[val_ids]
            sample_ids_train = sample_ids.loc[train_ids]

            _add_splits_to_result(X_pool, X_test, y_targets_pool, y_targets_test,
                                X_val, y_targets_val, sample_ids=sample_ids_train, sample_ids_val=sample_ids_val)

            logger.info(f'Size of the training set: {X_pool.shape[0]}')
            logger.info(f'Size of the validation set: {X_val.shape[0]}')
            logger.info(f'Size of the test set: {X_test.shape[0]}')
        else:
            # Random train-test split
            # Combine all targets into a single dataframe for splitting
            y_combined = pd.DataFrame(y_targets)
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y_combined, test_size=test_size, random_state=random_state
            )

            # Convert back to target dictionaries
            y_targets_pool = {target: y_pool[target] for target in config.target_columns}
            y_targets_test = {target: y_test[target] for target in config.target_columns}

            _add_splits_to_result(X_pool, X_test, y_targets_pool, y_targets_test,
                                sample_ids=sample_ids.loc[X_pool.index])
    else:
        # Fixed training IDs: entries that will always be kept in the training set
        X_fixed_train = X.loc[fixed_train_ids]
        y_targets_fixed_train = {target: y_targets[target].loc[fixed_train_ids] for target in config.target_columns}
        sample_ids_fixed_train = sample_ids.loc[fixed_train_ids]

        # Split remaining data
        X_remaining = X.drop(fixed_train_ids)
        y_remaining = pd.DataFrame({
            target: y_targets[target].drop(fixed_train_ids) for target in config.target_columns
        })
        y_remaining['sample_ids'] = sample_ids.drop(fixed_train_ids)

        X_pool, X_test, y_pool, y_test = train_test_split(
            X_remaining, y_remaining, test_size=test_size, random_state=random_state
        )

        # Convert to target dictionaries
        y_targets_pool = {target: y_pool[target] for target in config.target_columns}
        y_targets_test = {target: y_test[target] for target in config.target_columns}

        _add_splits_to_result(X_pool, X_test, y_targets_pool, y_targets_test)

        # Add fixed training set to result
        result.update({
            'X_fixed_train': X_fixed_train,
            'y_targets_fixed_train': y_targets_fixed_train,
            'sample_ids_fixed_train': sample_ids_fixed_train
        })

        logger.info(f'Size of the fixed training set: {X_fixed_train.shape[0]}')
        logger.info(f'Size of the pool: {X_remaining.shape[0]}')

    return result


def reformat_results(folder):

    def fill_None(test_scores):
        if isinstance(test_scores, dict):
            null_keys = [i for i in test_scores.keys() if len(test_scores[i])==0]
            nonnull_keys = [i for i in test_scores.keys() if len(test_scores[i])!=0]
            test_scores = pd.DataFrame({i: test_scores[i] for i in nonnull_keys})
            for key in null_keys:
                test_scores[key]=None
        return test_scores

    if pathlib.Path(f'{folder}/all_dat.pkl').is_file():
        with open(f'{folder}/all_dat.pkl','rb') as f:
            [size_old_val,ids,test_scores,val_scores] = pickle.load(f)
    elif pathlib.Path(f'{folder}/all_dat.pkl.tmp').is_file():
        with open(f'{folder}/all_dat.pkl.tmp','rb') as f:
            [size_old_val,ids,test_scores,val_scores] = pickle.load(f)

    else:
        with open(f'{folder}/val_scores.pkl','rb') as f:
            val_scores = pickle.load(f)

        with open(f'{folder}/test_scores.pkl','rb') as f:
            test_scores = pickle.load(f)

        with open(f'{folder}/ids.pkl','rb') as f:
            ids = pickle.load(f)

        with open(f'{folder}/size_old_val.pkl','rb') as f:
            size_old_val = pickle.load(f)



    dat_size = pd.DataFrame(size_old_val)
    dat_size.columns = ['val_size']
    tot_train_val_size = len(ids['train_val'])
    dat_size['val_ratio'] = dat_size['val_size']/tot_train_val_size
    dat_size['train_size'] =  tot_train_val_size - dat_size['val_size']
    dat_size['train_ratio'] = 1 - dat_size['val_ratio']
    test_scores['train_ratio'] = dat_size['train_ratio'] #.drop_duplicates()
    val_scores['train_ratio'] = dat_size['train_ratio'] #.drop_duplicates()

    # -- Fix a mini bug in my previous code ----
    diff = len(test_scores['train_ratio']) - len(test_scores['r2'])
    if diff >0:
        test_scores['train_ratio'] = test_scores['train_ratio'][:-diff]
        # with open(f'{folder}/all_dat.pkl'+'.tmp','wb') as f:
        #     pickle.dump([size_old_val,ids,test_scores,val_scores],f)

    diff = len(val_scores['train_ratio']) - len(val_scores['r2'])
    if diff >0:
        val_scores['train_ratio'] = val_scores['train_ratio'][:-diff]
    #     with open(f'{folder}/all_dat.pkl'+'.tmp','wb') as f:
    #         pickle.dump([size_old_val,ids,test_scores,val_scores],f)

    # ------------------------------------------

    test_scores = fill_None(test_scores)
    val_scores = fill_None(val_scores)

    ids['train_val'] = ids['old_val'] + ids['train_new_val']
    ids['train_val'].reverse()

    with open(f'{folder}/all_dat.pkl','wb') as f:
        pickle.dump([size_old_val,ids,test_scores,val_scores],f)



def get_indices_by_quantiles(s, n_sample):
    # Define the quantiles
    quantiles = np.linspace(0, 100, n_sample)

    # Initialize an empty list to store the indices
    indices = []

    # Iterate over the quantiles
    for q in quantiles:
        quantile_value = s.quantile(q / 100)  # Compute the quantile value
        indices.append(s[s <= quantile_value].sort_values(ascending=True).index[-1])  # Add the indices to the list

    return indices


def grow_data(
        model,
        X,y,
        X_test,y_test,
        file_out,
        X_val=None,y_val=None,
        n_iter: int = 50,
        subsample: float =1,
        batch_sizes: list = None,
        grow_criterion: str = 'max_err',
        ):

    # record id_list in each iteration
    id_train_iter = []
    train_size = []
    train_frac = []
    test_scores={'maes':[], 'rmse':[], 'r2':[], 'maes_m2':[], 'rmse_m2':[], 'r2_m2':[]}
    val_scores={'maes':[], 'rmse':[], 'r2':[], 'maes_m2':[], 'rmse_m2':[], 'r2_m2':[]}

    # pool ids
    pool = X.index.tolist()
    # total number
    ntot = len(pool)

    # define the number of samples to add in each iteration
    if batch_sizes is None:
        batch_size = int(len(pool) / n_iter)
        batch_sizes = [batch_size] * (n_iter + 1)
        logger.info(f'batch_size: {batch_size}')

    logger.info(f'# entries in pool: {ntot}')

    id_train = []
    # query by errors
    for n, batch_size in enumerate(batch_sizes):
        if n == 0:
            id_train = get_indices_by_quantiles(y, batch_size)

        # Update the pool by removing the samples in the training set
        pool = list(set(pool) - set(id_train))
        n_train = len(id_train)
        train_size.append(n_train)
        train_frac.append(n_train/ntot)

        # training
        X_train = X.loc[id_train]
        y_train = y.loc[id_train]
        X_pool = X.loc[pool]
        y_pool = y.loc[pool]

        if X_val is None:
            model.fit(X_train,y_train)
        elif X_val == 'X_pool':
            model.fit(X_train,y_train,val=(X_pool,y_pool))
        else:
            model.fit(X_train,y_train,val=(X_val,y_val))



        # test score
        y_pred = model.predict(X_test)
        maes_test = metrics.mean_absolute_error(y_test,y_pred)
        rmse_test = metrics.mean_squared_error(y_test,y_pred,squared=False)
        r2_test = metrics.r2_score(y_test,y_pred)
        test_scores['maes'].append(maes_test)
        test_scores['rmse'].append(rmse_test)
        test_scores['r2'].append(r2_test)


        # Get the prediction errors for the samples in pool
        y_pred = model.predict(X_pool)

        maes_val = metrics.mean_absolute_error(y_pool,y_pred)
        rmse_val = metrics.mean_squared_error(y_pool,y_pred,squared=False)
        r2_val = metrics.r2_score(y_pool,y_pred)
        val_scores['maes'].append(maes_val)
        val_scores['rmse'].append(rmse_val)
        val_scores['r2'].append(r2_val)

        logger.info('')
        logger.info(f'@ train_size: {n_train}, train_frac: {n_train/ntot:.3f}')
        logger.info(f'@ Val scores: maes={maes_val:.3f}, rmse={rmse_val:.3f}, r2={r2_val:.3f}')
        logger.info(f'@ Test scores: maes={maes_test:.3f}, rmse={rmse_test:.3f}, r2={r2_test:.3f}')
        logger.info('')

        with open(file_out,'wb') as f:
            pickle.dump([train_size,train_frac,id_train_iter,test_scores,val_scores],f)

        if n_train + batch_size > ntot:
            break

        # Select the samples to add in the next round of training
        # get the errors
        y_err = (y_pool-y_pred).abs()
        # subsample
        y_err = y_err.sample(
            frac=subsample,
            random_state=n
            )
        y_err = y_err.sort_values(ascending=True)

        if grow_criterion == 'max_err':
            new_id = y_err[-batch_size:].index.tolist()
        elif grow_criterion == 'min_err':
            new_id = y_err[:batch_size].index.tolist()
        else:
            raise ValueError(f"Unknown grow_criterion: {grow_criterion!r}")

        # add the new samples to the training set
        id_train.extend(new_id)
        id_train_iter.append(new_id)

    return train_size,train_frac,id_train_iter,test_scores,val_scores
