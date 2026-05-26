#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import random

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import math
import time
import pathlib


from uncertainty_Thermo import grow_random

# sum of disagreements
from uncertainty_Thermo import grow_QBC
from uncertainty_Thermo import grow_2QBC
from uncertainty_Thermo import grow_3QBC
from uncertainty_Thermo import grow_multiobj_all

# disagreements  + diversity
from uncertainty_Thermo import grow_EA_DeltaDiversity
from uncertainty_Thermo import grow_EA_3Obj_TwoTargets
from uncertainty_Thermo import grow_EA_4Obj_ThreeTargets

# NSGA-II disagreements 
from uncertainty_Thermo import grow_EA_2Obj_TwoTargets_QBC_ONLY
from uncertainty_Thermo import grow_EA_3Obj_ThreeTargets_QBC_ONLY

def split_XY(df):
    """
    Splits a dataframe into X (features) and Y (targets).

    Y columns:
        - Electrical Conductivity (S/cm)
        - Seebeck Coefficient (µV/K)
        - zT
        - Total Thermal Conductivity (W/mK)
    X contains all other columns.
    """

    y_cols = [
    "Electrical Conductivity (S/cm)",
    "Seebeck Coefficient (µV/K)",
    "zT",
    "Total Thermal Conductivity (W/mK)"]


    Y = df[y_cols].copy()
    X = df.drop(columns=y_cols).copy()

    return X, Y

from pathlib import Path
import re

def safe_filename(name):
    """
    Convert column names into safe file names.
    Removes characters that Windows interprets as paths or invalid symbols.
    """
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)  # invalid Windows filename chars
    name = name.replace(" ", "_")
    name = name.replace("(", "").replace(")", "")
    return name



def return_model(modelname,random_state):
    
    
    model = {}
    if modelname == 'xgb':
        return xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.25,
        reg_lambda=0.01,reg_alpha=0.1,
        subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=4 ,device="cpu"
        )
    
    if modelname == 'xgb_1tree':
        return xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.25,
        reg_lambda=0.01, #reg_alpha=0.1, # ibug requires that model_params['reg_alpha'] == 0
        # subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=1 ,device="cpu"
        )

    elif modelname == 'xgb_cla':
        return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.25,
        reg_lambda=0.01,reg_alpha=0.1,
        subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=4,device="cpu"
        )
    elif modelname == 'rf':
        return RandomForestRegressor(
        n_estimators=100, max_features=1/3, n_jobs=-1, random_state=random_state
        )

    else:
        raise ValueError(f'Unknown model: {modelname}')



print('Dependencies imported successfully!')

#%%
# REQUIRE USER INPUT---------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(
    description='Grow datasets with different growing methods.'
    )

parser.add_argument('--growingCriteria', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--outputDir', type=str, required=True)
parser.add_argument('--stepFrac', type=float, required=False, default=0.01) #CHANGE
parser.add_argument('--trainFracStop', type=float, required=False, default=1.0)
parser.add_argument('--randomSeed', type=int, required=False, default=0)


growingCriteria = parser.parse_args().growingCriteria
dataset = parser.parse_args().dataset
target = parser.parse_args().target
outputDir = parser.parse_args().outputDir
fltStepFrac = parser.parse_args().stepFrac
fltTrainFracStop = parser.parse_args().trainFracStop
intRandomSeed = parser.parse_args().randomSeed


# %%
# IMPORT MODELS--------------------------------------------------------------------------------------------------------
# set random state
random_state = 0

# make a dictionary to hold the models
model = {}

# loop through the models and add them to the dictionary
for modelname in ['xgb','rf']:
    model[modelname] = return_model(modelname, random_state)
    # print(modelname, ' imported successfully: ', model[modelname])

print('Models imported successfully!')
print("GrowingCriteria is",growingCriteria)

print(intRandomSeed)



TE_COLMAP = {
    "electric_cond": "Electrical Conductivity (S/cm)",
    "seebeck_coef":  "Seebeck Coefficient (µV/K)",
    "zT":            "zT",
    "thermal_cond":  "Total Thermal Conductivity (W/mK)",

    # optional shorthands you used in combos:
    "el":   "Electrical Conductivity (S/cm)",
    "seeb": "Seebeck Coefficient (µV/K)",
    "ThC":  "Total Thermal Conductivity (W/mK)",
}

CRIT_TO_NTARGETS = {
    # 1 outcome
    "grow_QBC": 1,
    "grow_EA_DeltaDiversity": 1,

    # 2 outcomes
    "grow_2QBC": 2,
    "grow_EA_3Obj_TwoTargets": 2,
    "grow_EA_2Obj_TwoTargets_QBC_ONLY": 2,

    # 3 outcomes
    "grow_3QBC": 3,
    "grow_EA_4Obj_ThreeTargets": 3,
    "grow_EA_3Obj_ThreeTargets_QBC_ONLY": 3,

    # all outcomes
    "grow_random": 4,
    "grow_multiobj_all": 4,
}


def resolve_te_targets(growingCriteria: str, target: str):
    """
    Returns:
      - for 1 target criteria: (target1,)
      - for 2 target criteria: (target1, target2)
      - for 3 target criteria: (target1, target2, target3)
      - for 4 target criteria: (target_cols_list,) OR just the full list (your choice)
    where each target is the FULL TE column name.
    """

    if growingCriteria not in CRIT_TO_NTARGETS:
        raise ValueError(f"Unknown growingCriteria '{growingCriteria}' in CRIT_TO_NTARGETS")

    n = CRIT_TO_NTARGETS[growingCriteria]

    t = target.strip()

    # --- all outcomes case ---
    if n == 4:
        if t != "all":
            raise ValueError(f"For growingCriteria='{growingCriteria}', target must be 'all', got '{target}'")
        target_cols = [
            "Electrical Conductivity (S/cm)",
            "Seebeck Coefficient (µV/K)",
            "zT",
            "Total Thermal Conductivity (W/mK)",
        ]
        return tuple(target_cols)

    # --- 1/2/3 outcomes: parse tokens ---
    # Expect underscore-separated codes like:
    #   electric_cond
    #   el_seeb
    #   el_seeb_zT
    parts = t.split("_")

    # Special case: if someone passes "thermal_cond" it becomes ["thermal","cond"] -> fix it
    # and same for "electric_cond", "seebeck_coef"
    # Solution: first try direct lookup; if not found, then treat underscores as separators.
    if n == 1:
        if t not in TE_COLMAP:
            raise ValueError(f"Unknown target code '{t}'. Expected one of: {list(TE_COLMAP.keys())}")
        return (TE_COLMAP[t],)

    # For n>1 we expect combos like el_seeb, el_seeb_zT, etc.
    # We'll parse by recognizing valid tokens from TE_COLMAP.
    # Example: "zT_ThC" -> ["zT","ThC"] works.
    combo_tokens = parts

    if len(combo_tokens) != n:
        raise ValueError(
            f"Target '{target}' implies {len(combo_tokens)} tokens, but growingCriteria '{growingCriteria}' expects {n} targets."
        )

    resolved = []
    for tok in combo_tokens:
        if tok not in TE_COLMAP:
            raise ValueError(f"Unknown token '{tok}' in target='{target}'. Allowed: {list(TE_COLMAP.keys())}")
        resolved.append(TE_COLMAP[tok])

    return tuple(resolved)


if growingCriteria in ("grow_2QBC", "grow_EA_3Obj_TwoTargets", "grow_EA_2Obj_TwoTargets_QBC_ONLY"):
    target1, target2 = resolve_te_targets(growingCriteria, target)

elif growingCriteria in ("grow_3QBC", "grow_EA_4Obj_ThreeTargets", "grow_EA_3Obj_ThreeTargets_QBC_ONLY"):
    target1, target2, target3 = resolve_te_targets(growingCriteria, target)

elif growingCriteria in ("grow_QBC", "grow_EA_DeltaDiversity"):
    (target1,) = resolve_te_targets(growingCriteria, target)

elif growingCriteria in ("grow_random", "grow_multiobj_all"):
    target_cols = list(resolve_te_targets(growingCriteria, target))


df_clean=pd.read_csv(r'CLEANED_DATA.csv')
print(df_clean.head())
df_clean = df_clean.loc[:, ~df_clean.columns.str.contains("^Unnamed")]
print(df_clean.head())
# ---- 1) Split 10% as test set ----
df_trainval, df_test = train_test_split(
    df_clean,
    test_size=0.10,
    random_state=42,
    shuffle=True
)


xTrainVal,yTrainVal=split_XY(df_trainval)
xTest,yTest=split_XY(df_test)


# Output folder
out_dir = Path("split_data_csv")
out_dir.mkdir(parents=True, exist_ok=True)

# Save X data
xTrainVal.to_csv(out_dir / "xTrainVal.csv", index=True)
xTest.to_csv(out_dir / "xTest.csv", index=True)

# Save each yTrainVal column separately
for col in yTrainVal.columns:
    filename = f"yTrainVal_{safe_filename(col)}.csv"
    yTrainVal[[col]].to_csv(out_dir / filename, index=True)

# Save each yTest column separately
for col in yTest.columns:
    filename = f"yTest_{safe_filename(col)}.csv"
    yTest[[col]].to_csv(out_dir / filename, index=True)




if growingCriteria == 'grow_random':
    grow_random(model['rf'], 
                 model['xgb'],
                xTrainVal, yTrainVal,
                xTest, yTest, 
                strSaveDir = outputDir,
                strSaveName = dataset + '_' + target + '_random',
                strModel1Name = 'rf',
                strModel2Name = 'xgb',
                fltStepFrac = fltStepFrac,
                intRandomSeed = intRandomSeed)
    
elif growingCriteria=='grow_QBC':
    grow_QBC(model['rf'], model['xgb'],
             xTrainVal, yTrainVal,
             xTest, yTest,
             target=target1,
             strSaveDir = outputDir,
             strSaveName =dataset + '_' + target +'growing_QBC',
             strModel1Name = 'rf',
             strModel2Name = 'xgb',
             fltStepFrac = fltStepFrac,
             intRandomSeed = intRandomSeed)


# QBC + Diversity using NSGA-II 
elif growingCriteria=='grow_EA_DeltaDiversity':
    grow_EA_DeltaDiversity(
        model['rf'], model['xgb'],
        xTrainVal, yTrainVal,
        xTest, yTest,
        target=target1,
        strSaveDir=outputDir,
        strSaveName=dataset + '_' + target +'growing_EA_1out_div',
        strModel1Name='rf',
        strModel2Name='xgb',
        fltStepFrac=fltStepFrac,
        intRandomSeed=intRandomSeed
    )

# Two  outcomes QBC 

elif growingCriteria=='grow_2QBC':
    grow_2QBC(model['rf'], model['xgb'],
             xTrainVal, yTrainVal,
             xTest, yTest,
             target1=target1,
             target2=target2,           
             w1=1.0, w2=1.0,          
             strSaveDir=outputDir,
             strSaveName=dataset + '_' + target +'growing_2QBC',
             strModel1Name='rf',
             strModel2Name='xgb',
             fltStepFrac=fltStepFrac,
             intRandomSeed=intRandomSeed,
             n_round=6,
             n_train_ratio=4)
    
# QBC with NSGA-II . No diversity in feature space

elif growingCriteria=='grow_EA_2Obj_TwoTargets_QBC_ONLY':
    grow_EA_2Obj_TwoTargets_QBC_ONLY(
            model['rf'], model['xgb'],
            xTrainVal, yTrainVal,
            xTest, yTest,
            target1=target1,
            target2=target2,
            strSaveDir=outputDir,
            strSaveName=dataset + '_' + target +'growing_EA_2obj_qbc_only',
            strModel1Name='rf',
            strModel2Name='xgb',
            fltStepFrac=fltStepFrac,
            intRandomSeed=intRandomSeed,
            pop_size=120,
            n_gen=60,
            p_cx=0.9,
            p_mut=0.25)
    


# Two  outcomes QBC + Diversity  with NSGA-II 

elif growingCriteria=='grow_EA_3Obj_TwoTargets':

    grow_EA_3Obj_TwoTargets(
    model['rf'], model['xgb'],
    xTrainVal, yTrainVal,
    xTest, yTest,
    target1=target1,
    target2=target2,
    strSaveDir=outputDir,
    strSaveName=dataset + '_' + target +'growing_EA_2out_div',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=fltStepFrac,
    intRandomSeed=intRandomSeed,
    do_scale_global=True,
    normalize_global=True,
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    w_train=0.5,
    w_within=0.5)




# Three outcomes QBC 

elif growingCriteria=='grow_3QBC':
    grow_3QBC(model['rf'], model['xgb'],
             xTrainVal, yTrainVal,
             xTest, yTest,
             target1=target1,
             target2=target2,
             target3=target3,              
             w1=1.0, w2=1.0, w3=1.0,     
             strSaveDir=outputDir,
             strSaveName=dataset + '_' + target +'growing_3QBC',
             strModel1Name='rf',
             strModel2Name='xgb',
             fltStepFrac=fltStepFrac,
             intRandomSeed=intRandomSeed,
             n_round=6,
             n_train_ratio=4)


# Three outcomes QBC + Diversity  with NSGA-II 


elif growingCriteria=='grow_EA_4Obj_ThreeTargets':
    grow_EA_4Obj_ThreeTargets(
        model['rf'], model['xgb'],
        xTrainVal, yTrainVal,
        xTest, yTest,
        target1=target1,
        target2=target2,
        target3=target3,
        strSaveDir=outputDir,
        strSaveName=dataset + '_' + target +'growing_EA_3out_div',
        strModel1Name='rf',
        strModel2Name='xgb',
        fltStepFrac=fltStepFrac,
        intRandomSeed=intRandomSeed,
        do_scale_global=True,
        normalize_global=True,
        pop_size=120,
        n_gen=60,
        p_cx=0.9,
        p_mut=0.25,
        w_train=0.5,
        w_within=0.5,
    )

# Three outcomes QBC with NSGA-II . No diversity in feature space

elif growingCriteria=='grow_EA_3Obj_ThreeTargets_QBC_ONLY':
    grow_EA_3Obj_ThreeTargets_QBC_ONLY(
        model['rf'], model['xgb'],
        xTrainVal, yTrainVal,
        xTest, yTest,
        target1=target1,
        target2=target2,
        target3=target3,
        strSaveDir=outputDir,
        strSaveName=dataset + '_' + target +'growing_EA_3obj_qbc_only',
        strModel1Name='rf',
        strModel2Name='xgb',
        fltStepFrac=fltStepFrac,
        intRandomSeed=intRandomSeed,
        pop_size=120,
        n_gen=60,
        p_cx=0.9,
        p_mut=0.25)

# Four outcomes QBC with NSGA-II 


elif growingCriteria=='grow_multiobj_all':
    grow_multiobj_all(model['rf'], model['xgb'],
                        xTrainVal, yTrainVal,
                        xTest, yTest,
                        strSaveDir=outputDir,
                        strSaveName=dataset + '_' + target +'growing_QBC_all',
                        strModel1Name='rf',
                        strModel2Name='xgb',
                        fltStepFrac=fltStepFrac,
                        intRandomSeed=intRandomSeed,
                        n_round=6,
                        n_train_ratio=4,
                        weights=None)
