import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize
import time
from sklearn import metrics

n_round = 6     # as in your other scripts
# n_train_ratio must also exist; if not, define e.g.:
n_train_ratio = 2
# =========================
# Utilities
# =========================
def minMaxStandarize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


def get_scores(model,X_train,y_train,X_test,y_test, X_val=None, y_val=None):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if len(X_val)==0:
        y_pred_val=-1
    else:
        y_pred_val=model.predict(X_val)
    maes = metrics.mean_absolute_error(y_test,y_pred)
    rmse = metrics.root_mean_squared_error(y_test,y_pred)  #   mean_squared_error(y_test,y_pred,squared=False)
    r2 = metrics.r2_score(y_test,y_pred)
    print(f'Test scores: MAE={maes:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}')
    if (X_val is not None) and (y_val is not None):
        y_pred = model.predict(X_val)
        maes_val = metrics.mean_absolute_error(y_val,y_pred)
        rmse_val = metrics.root_mean_squared_error(y_val,y_pred) #   mean_squared_error(y_test,y_pred,squared=False)
        r2_val = metrics.r2_score(y_val,y_pred)
        print(f'Val scores: MAE={maes_val:.3f}, RMSE={rmse_val:.3f}, R2={r2_val:.3f}')
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        return maes, rmse, r2, maes_val, rmse_val, r2_val
    else:
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        return y_pred_val,maes, rmse, r2


def splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices):

    target_cols =[
    "Electrical Conductivity (S/cm)",
    "Seebeck Coefficient (µV/K)",
    "zT",
    "Total Thermal Conductivity (W/mK)"]

    yTargets = yTrainVal[target_cols]

    xTrain = xTrainVal.loc[lstTrainIndices]
    yTrain = yTargets.loc[lstTrainIndices]

    xVal = xTrainVal.drop(lstTrainIndices)
    yVal = yTargets.drop(lstTrainIndices)

    return xTrain, xVal, yTrain, yVal


def grow_random(model1, 
                model2,
                xTrainVal, yTrainVal,
                xTest, yTest, 
                strSaveDir = '',
                strSaveName = 'growing_random',
                strModel1Name = 'rf',
                strModel2Name = 'xgb',
                fltStepFrac = 0.01,
                intRandomSeed = 0):

    # ---- TARGET COLUMNS ----
    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]

    # ---- SAFETY CHECKS ----
    for col in target_cols:
        assert col in yTrainVal.columns, f"Column '{col}' missing from yTrainVal"
        assert col in yTest.columns,     f"Column '{col}' missing from yTest"

    # normalize save path
    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    print('RANDOM SEED IN THE GROW RANDOM FUNCTION')
    print(intRandomSeed)

    # ---- INIT RESULT TABLES ----
    dfGrowing_random_model1 = pd.DataFrame()
    dfGrowing_random_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize      = int(len(xTrainVal) * fltStepFrac)

    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    # ---- FIXED TEST TARGETS ----
    yEC_test   = yTest["Electrical Conductivity (S/cm)"]
    yS_test    = yTest["Seebeck Coefficient (µV/K)"]
    yZT_test   = yTest["zT"]
    yKtot_test = yTest["Total Thermal Conductivity (W/mK)"]

    # ==================================================================
    #                           MAIN LOOP
    # ==================================================================
    while intTempTrainSize < (len(xTrainVal) + 1):

        timeStart = time.time()

        yTrainVal_targets = yTrainVal[target_cols]

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal, yTrainVal_targets, lstTrainIndices
        )

        yEC_train   = yTrain_temp["Electrical Conductivity (S/cm)"]
        yS_train    = yTrain_temp["Seebeck Coefficient (µV/K)"]
        yZT_train   = yTrain_temp["zT"]
        yKtot_train = yTrain_temp["Total Thermal Conductivity (W/mK)"]

        # ---- SANITY INDEX CHECKS ----
        assert len(set(xTrainVal.index) & set(xTest.index)) == 0
        assert len(set(xTrain_temp.index) & set(xVal_temp.index)) == 0
        assert len(set(xTrain_temp.index) & set(xTest.index)) == 0

        print('\n------------------------------------------------------------')
        print('Training set size:', len(xTrain_temp))
        print('Training %:', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # =============================================================
        #                         MODEL 1
        # =============================================================

        print("---- MODEL1: ELECTRICAL CONDUCTIVITY ----")
        _, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(
            model1, xTrain_temp, yEC_train, xTest, yEC_test, xVal_temp
        )

        print("---- MODEL1: SEEBECK COEFFICIENT ----")
        _, s_mae_1, s_rmse_1, s_r2_1 = get_scores(
            model1, xTrain_temp, yS_train, xTest, yS_test, xVal_temp
        )

        print("---- MODEL1: zT ----")
        _, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(
            model1, xTrain_temp, yZT_train, xTest, yZT_test, xVal_temp
        )

        print("---- MODEL1: TOTAL THERMAL CONDUCTIVITY ----")
        _, k_mae_1, k_rmse_1, k_r2_1 = get_scores(
            model1, xTrain_temp, yKtot_train, xTest, yKtot_test, xVal_temp
        )

        # =============================================================
        #                         MODEL 2
        # =============================================================

        print("---- MODEL2: ELECTRICAL CONDUCTIVITY ----")
        _, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(
            model2, xTrain_temp, yEC_train, xTest, yEC_test, xVal_temp
        )

        print("---- MODEL2: SEEBECK COEFFICIENT ----")
        _, s_mae_2, s_rmse_2, s_r2_2 = get_scores(
            model2, xTrain_temp, yS_train, xTest, yS_test, xVal_temp
        )

        print("---- MODEL2: zT ----")
        _, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(
            model2, xTrain_temp, yZT_train, xTest, yZT_test, xVal_temp
        )

        print("---- MODEL2: TOTAL THERMAL CONDUCTIVITY ----")
        _, k_mae_2, k_rmse_2, k_r2_2 = get_scores(
            model2, xTrain_temp, yKtot_train, xTest, yKtot_test, xVal_temp
        )

        # =============================================================
        #                        STORE RESULTS
        # =============================================================

        dfGrowing_random_model1 = pd.concat([
            dfGrowing_random_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 2)],

                'mae_elec_cond':   [round(ec_mae_1,   n_round)],
                'rmse_elec_cond':  [round(ec_rmse_1, n_round)],
                'r2_elec_cond':    [round(ec_r2_1,   n_round)],

                'mae_seebeck':     [round(s_mae_1,   n_round)],
                'rmse_seebeck':    [round(s_rmse_1, n_round)],
                'r2_seebeck':      [round(s_r2_1,   n_round)],

                'mae_zt':          [round(zt_mae_1,   n_round)],
                'rmse_zt':         [round(zt_rmse_1, n_round)],
                'r2_zt':           [round(zt_r2_1,   n_round)],

                'mae_k_total':     [round(k_mae_1,   n_round)],
                'rmse_k_total':    [round(k_rmse_1, n_round)],
                'r2_k_total':      [round(k_r2_1,   n_round)],

                'train_size':      [len(xTrain_temp)]
            })
        ], ignore_index=True)

        dfGrowing_random_model2 = pd.concat([
            dfGrowing_random_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 2)],

                'mae_elec_cond':   [round(ec_mae_2,   n_round)],
                'rmse_elec_cond':  [round(ec_rmse_2, n_round)],
                'r2_elec_cond':    [round(ec_r2_2,   n_round)],

                'mae_seebeck':     [round(s_mae_2,   n_round)],
                'rmse_seebeck':    [round(s_rmse_2, n_round)],
                'r2_seebeck':      [round(s_r2_2,   n_round)],

                'mae_zt':          [round(zt_mae_2,   n_round)],
                'rmse_zt':         [round(zt_rmse_2, n_round)],
                'r2_zt':           [round(zt_r2_2,   n_round)],

                'mae_k_total':     [round(k_mae_2,   n_round)],
                'rmse_k_total':    [round(k_rmse_2, n_round)],
                'r2_k_total':      [round(k_r2_2,   n_round)],

                'train_size':      [len(xTrain_temp)]
            })
        ], ignore_index=True)

        # ---- SAVE PROGRESS ----
        dfGrowing_random_model1.to_csv(
            f"{strSaveDir}{strSaveName}_{strModel1Name}{intRandomSeed}.csv",
            index=False
        )
        dfGrowing_random_model2.to_csv(
            f"{strSaveDir}{strSaveName}_{strModel2Name}{intRandomSeed}.csv",
            index=False
        )

        pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
            f"{strSaveDir}{strSaveName}_trainIndices.csv",
            index=False
        )

        # ---- GROW TRAINING SET ----
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        if intTempTrainSize + intStepSize > len(xTrainVal):
            intStepSize = len(xTrainVal) - intTempTrainSize

        lstTempIndices = np.random.choice(
            xVal_temp.index,
            intStepSize,
            replace=False
        ).tolist()

        lstTrainIndices += lstTempIndices
        intTempTrainSize = len(lstTrainIndices)

        print("Loop time:", round(time.time() - timeStart, 2), "s")

    # ---- FINAL SAVE ----
    dfGrowing_random_model1.to_csv(
        f"{strSaveDir}{strSaveName}_{strModel1Name}{intRandomSeed}.csv",
        index=False
    )
    dfGrowing_random_model2.to_csv(
        f"{strSaveDir}{strSaveName}_{strModel2Name}{intRandomSeed}.csv",
        index=False
    )

    pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
        f"{strSaveDir}{strSaveName}_trainIndices.csv",
        index=False
    )

    return



def grow_QBC(model1, model2,
             xTrainVal, yTrainVal,
             xTest, yTest,
             target,
             strSaveDir = '',
             strSaveName ='growing_QBC',
             strModel1Name = 'rf',
             strModel2Name = 'xgb',
             fltStepFrac = 0.01,
             intRandomSeed = 0):
    """
    QBC active learning growth for thermoelectric outcomes.

    Targets (columns of yTrainVal / yTest):
        - 'Electrical Conductivity (S/cm)'
        - 'Seebeck Coefficient (µV/K)'
        - 'zT'
        - 'Total Thermal Conductivity (W/mK)'

    `target` controls which of the four is used to compute QBC disagreement
    and must be one of the strings above.
    """

    import random
    import numpy as np
    import pandas as pd
    import time

    # ---- TARGET COLUMNS ----
    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]

    # safety checks
    for col in target_cols:
        assert col in yTrainVal.columns, f"Column '{col}' missing from yTrainVal"
        assert col in yTest.columns,     f"Column '{col}' missing from yTest"

    if target not in target_cols:
        raise ValueError(
            "target must be one of: "
            "'Electrical Conductivity (S/cm)', "
            "'Seebeck Coefficient (µV/K)', "
            "'zT', "
            "'Total Thermal Conductivity (W/mK)'"
        )

    # normalize save path
    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    dfGrowing_QBC_model1 = pd.DataFrame()
    dfGrowing_QBC_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize      = int(len(xTrainVal) * fltStepFrac)

    # initial random training indices
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    # fixed test targets
    yEC_test   = yTest["Electrical Conductivity (S/cm)"]
    yS_test    = yTest["Seebeck Coefficient (µV/K)"]
    yZT_test   = yTest["zT"]
    yKtot_test = yTest["Total Thermal Conductivity (W/mK)"]

    # ========================= MAIN LOOP =========================
    while intTempTrainSize < (len(xTrainVal) + 1):

        timeStart = time.time()

        # work only with target columns
        yTrainVal_targets = yTrainVal[target_cols]

        # split into train / val using your splitTrainVal (index-respecting)
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal, yTrainVal_targets, lstTrainIndices
        )

        print('\n------------------------------------------------------------')
        print('Training set size:', len(xTrain_temp))
        print('Training %:       ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # unpack train targets
        yEC_train   = yTrain_temp["Electrical Conductivity (S/cm)"]
        yS_train    = yTrain_temp["Seebeck Coefficient (µV/K)"]
        yZT_train   = yTrain_temp["zT"]
        yKtot_train = yTrain_temp["Total Thermal Conductivity (W/mK)"]

        # ================= MODEL 1 SCORES =================
        print("---- MODEL1: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(
            model1,
            xTrain_temp, yEC_train,
            xTest,      yEC_test,
            X_val=xVal_temp
        )

        print("---- MODEL1: SEEBECK COEFFICIENT ----")
        s_pred_val1, s_mae_1, s_rmse_1, s_r2_1 = get_scores(
            model1,
            xTrain_temp, yS_train,
            xTest,      yS_test,
            X_val=xVal_temp
        )

        print("---- MODEL1: zT ----")
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(
            model1,
            xTrain_temp, yZT_train,
            xTest,      yZT_test,
            X_val=xVal_temp
        )

        print("---- MODEL1: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val1, k_mae_1, k_rmse_1, k_r2_1 = get_scores(
            model1,
            xTrain_temp, yKtot_train,
            xTest,      yKtot_test,
            X_val=xVal_temp
        )

        # ================= MODEL 2 SCORES =================
        print("---- MODEL2: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(
            model2,
            xTrain_temp, yEC_train,
            xTest,      yEC_test,
            X_val=xVal_temp
        )

        print("---- MODEL2: SEEBECK COEFFICIENT ----")
        s_pred_val2, s_mae_2, s_rmse_2, s_r2_2 = get_scores(
            model2,
            xTrain_temp, yS_train,
            xTest,      yS_test,
            X_val=xVal_temp
        )

        print("---- MODEL2: zT ----")
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(
            model2,
            xTrain_temp, yZT_train,
            xTest,      yZT_test,
            X_val=xVal_temp
        )

        print("---- MODEL2: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val2, k_mae_2, k_rmse_2, k_r2_2 = get_scores(
            model2,
            xTrain_temp, yKtot_train,
            xTest,      yKtot_test,
            X_val=xVal_temp
        )

        # ================= STORE RESULTS =================
        dfGrowing_QBC_model1 = pd.concat(
            [
                dfGrowing_QBC_model1,
                pd.DataFrame({
                    'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                    'mae_elec_cond':   [round(ec_mae_1,   n_round)],
                    'rmse_elec_cond':  [round(ec_rmse_1, n_round)],
                    'r2_elec_cond':    [round(ec_r2_1,   n_round)],

                    'mae_seebeck':     [round(s_mae_1,   n_round)],
                    'rmse_seebeck':    [round(s_rmse_1, n_round)],
                    'r2_seebeck':      [round(s_r2_1,   n_round)],

                    'mae_zt':          [round(zt_mae_1,   n_round)],
                    'rmse_zt':         [round(zt_rmse_1, n_round)],
                    'r2_zt':           [round(zt_r2_1,   n_round)],

                    'mae_k_total':     [round(k_mae_1,   n_round)],
                    'rmse_k_total':    [round(k_rmse_1, n_round)],
                    'r2_k_total':      [round(k_r2_1,   n_round)],

                    'train_size':      [len(xTrain_temp)]
                })
            ],
            ignore_index=True
        )

        dfGrowing_QBC_model2 = pd.concat(
            [
                dfGrowing_QBC_model2,
                pd.DataFrame({
                    'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                    'mae_elec_cond':   [round(ec_mae_2,   n_round)],
                    'rmse_elec_cond':  [round(ec_rmse_2, n_round)],
                    'r2_elec_cond':    [round(ec_r2_2,   n_round)],

                    'mae_seebeck':     [round(s_mae_2,   n_round)],
                    'rmse_seebeck':    [round(s_rmse_2, n_round)],
                    'r2_seebeck':      [round(s_r2_2,   n_round)],

                    'mae_zt':          [round(zt_mae_2,   n_round)],
                    'rmse_zt':         [round(zt_rmse_2, n_round)],
                    'r2_zt':           [round(zt_r2_2,   n_round)],

                    'mae_k_total':     [round(k_mae_2,   n_round)],
                    'rmse_k_total':    [round(k_rmse_2, n_round)],
                    'r2_k_total':      [round(k_r2_2,   n_round)],

                    'train_size':      [len(xTrain_temp)]
                })
            ],
            ignore_index=True
        )

        # ================= QBC SELECTION =================
        # choose which predictions drive disagreement
        if target == "Electrical Conductivity (S/cm)":
            val_pred1 = ec_pred_val1
            val_pred2 = ec_pred_val2
        elif target == "Seebeck Coefficient (µV/K)":
            val_pred1 = s_pred_val1
            val_pred2 = s_pred_val2
        elif target == "zT":
            val_pred1 = zt_pred_val1
            val_pred2 = zt_pred_val2
        elif target == "Total Thermal Conductivity (W/mK)":
            val_pred1 = k_pred_val1
            val_pred2 = k_pred_val2

        # disagreement on validation set
        yVal_temp_diff = np.abs(val_pred1 - val_pred2)
        yVal_temp_diff = pd.Series(yVal_temp_diff, index=yVal_temp.index)

        # pick intStepSize most-disagreeing points
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # ---- STOP IF FULL TRAINVAL USED ----
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        # ---- UPDATE TRAIN SET ----
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n------------------------------------------------------------')

    # ================= FINAL SAVE =================
    dfGrowing_QBC_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed)+'.csv',
        index=False
    )
    dfGrowing_QBC_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed)+ '.csv',
        index=False
    )

    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    dfTrainIndices.to_csv(
        strSaveDir + strSaveName + str(intRandomSeed)+'_trainIndices.csv',
        index=False
    )




    return



def grow_2QBC(model1, model2,
             xTrainVal, yTrainVal,
             xTest, yTest,
             target1,
             target2=None,           # <-- NEW: second disagreement target (optional)
             w1=1.0, w2=1.0,          # <-- NEW: weights for sum (optional)
             strSaveDir='',
             strSaveName='growing_QBC',
             strModel1Name='rf',
             strModel2Name='xgb',
             fltStepFrac=0.01,
             intRandomSeed=0,
             n_round=6,
             n_train_ratio=4):
    """
    QBC active learning growth for thermoelectric outcomes.

    Targets (columns of yTrainVal / yTest):
        - 'Electrical Conductivity (S/cm)'
        - 'Seebeck Coefficient (µV/K)'
        - 'zT'
        - 'Total Thermal Conductivity (W/mK)'

    Disagreement score used to acquire new points:
        - If target2 is None:
              score = minMaxStandarize(|pred1(target1) - pred2(target1)|)
        - If target2 is provided:
              score = w1*minMaxStandarize(|pred1(target1)-pred2(target1)|)
                    + w2*minMaxStandarize(|pred1(target2)-pred2(target2)|)

    Notes:
        - This function assumes you already have:
            * splitTrainVal(X, Y, lstTrainIndices) -> (X_train, X_val, Y_train, Y_val)
            * get_scores(model, X_train, y_train, X_test, y_test, X_val=...) ->
                  (y_pred_val, mae, rmse, r2)
    """

    # ---- TARGET COLUMNS ----
    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]

    # safety checks
    for col in target_cols:
        assert col in yTrainVal.columns, f"Column '{col}' missing from yTrainVal"
        assert col in yTest.columns,     f"Column '{col}' missing from yTest"

    if target1 not in target_cols:
        raise ValueError(
            "target1 must be one of: "
            "'Electrical Conductivity (S/cm)', "
            "'Seebeck Coefficient (µV/K)', "
            "'zT', "
            "'Total Thermal Conductivity (W/mK)'"
        )
    if target2 is not None and target2 not in target_cols:
        raise ValueError(
            "target2 must be one of: "
            "'Electrical Conductivity (S/cm)', "
            "'Seebeck Coefficient (µV/K)', "
            "'zT', "
            "'Total Thermal Conductivity (W/mK)'"
        )

    # normalize save path
    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    dfGrowing_QBC_model1 = pd.DataFrame()
    dfGrowing_QBC_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize      = int(len(xTrainVal) * fltStepFrac)

    # guard against tiny fractions
    intTempTrainSize = max(1, intTempTrainSize)
    intStepSize      = max(1, intStepSizestepSize if (R:=0) else intStepSize)  # no-op but keeps style; safe
    # (If you dislike this line, remove it; intStepSize already set above.)

    # initial random training indices
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    # fixed test targets
    yEC_test   = yTest["Electrical Conductivity (S/cm)"]
    yS_test    = yTest["Seebeck Coefficient (µV/K)"]
    yZT_test   = yTest["zT"]
    yKtot_test = yTest["Total Thermal Conductivity (W/mK)"]

    # ========================= MAIN LOOP =========================
    while intTempTrainSize < (len(xTrainVal) + 1):

        timeStart = time.time()

        # work only with target columns
        yTrainVal_targets = yTrainVal[target_cols]

        # split into train / val using your splitTrainVal (index-respecting)
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal, yTrainVal_targets, lstTrainIndices
        )

        print('\n------------------------------------------------------------')
        print('Training set size:', len(xTrain_temp))
        print('Training %:       ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # unpack train targets
        yEC_train   = yTrain_temp["Electrical Conductivity (S/cm)"]
        yS_train    = yTrain_temp["Seebeck Coefficient (µV/K)"]
        yZT_train   = yTrain_temp["zT"]
        yKtot_train = yTrain_temp["Total Thermal Conductivity (W/mK)"]

        # ================= MODEL 1 SCORES =================
        print("---- MODEL1: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(
            model1,
            xTrain_temp, yEC_train,
            xTest,      yEC_test,
            X_val=xVal_temp
        )

        print("---- MODEL1: SEEBECK COEFFICIENT ----")
        s_pred_val1, s_mae_1, s_rmse_1, s_r2_1 = get_scores(
            model1,
            xTrain_temp, yS_train,
            xTest,      yS_test,
            X_val=xVal_temp
        )

        print("---- MODEL1: zT ----")
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(
            model1,
            xTrain_temp, yZT_train,
            xTest,      yZT_test,
            X_val=xVal_temp
        )

        print("---- MODEL1: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val1, k_mae_1, k_rmse_1, k_r2_1 = get_scores(
            model1,
            xTrain_temp, yKtot_train,
            xTest,      yKtot_test,
            X_val=xVal_temp
        )

        # ================= MODEL 2 SCORES =================
        print("---- MODEL2: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(
            model2,
            xTrain_temp, yEC_train,
            xTest,      yEC_test,
            X_val=xVal_temp
        )

        print("---- MODEL2: SEEBECK COEFFICIENT ----")
        s_pred_val2, s_mae_2, s_rmse_2, s_r2_2 = get_scores(
            model2,
            xTrain_temp, yS_train,
            xTest,      yS_test,
            X_val=xVal_temp
        )

        print("---- MODEL2: zT ----")
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(
            model2,
            xTrain_temp, yZT_train,
            xTest,      yZT_test,
            X_val=xVal_temp
        )

        print("---- MODEL2: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val2, k_mae_2, k_rmse_2, k_r2_2 = get_scores(
            model2,
            xTrain_temp, yKtot_train,
            xTest,      yKtot_test,
            X_val=xVal_temp
        )

        # ================= STORE RESULTS =================
        dfGrowing_QBC_model1 = pd.concat(
            [
                dfGrowing_QBC_model1,
                pd.DataFrame({
                    'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                    'mae_elec_cond':   [round(ec_mae_1,   n_round)],
                    'rmse_elec_cond':  [round(ec_rmse_1,  n_round)],
                    'r2_elec_cond':    [round(ec_r2_1,    n_round)],

                    'mae_seebeck':     [round(s_mae_1,    n_round)],
                    'rmse_seebeck':    [round(s_rmse_1,   n_round)],
                    'r2_seebeck':      [round(s_r2_1,     n_round)],

                    'mae_zt':          [round(zt_mae_1,   n_round)],
                    'rmse_zt':         [round(zt_rmse_1,  n_round)],
                    'r2_zt':           [round(zt_r2_1,    n_round)],

                    'mae_k_total':     [round(k_mae_1,    n_round)],
                    'rmse_k_total':    [round(k_rmse_1,   n_round)],
                    'r2_k_total':      [round(k_r2_1,     n_round)],

                    'train_size':      [len(xTrain_temp)]
                })
            ],
            ignore_index=True
        )

        dfGrowing_QBC_model2 = pd.concat(
            [
                dfGrowing_QBC_model2,
                pd.DataFrame({
                    'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                    'mae_elec_cond':   [round(ec_mae_2,   n_round)],
                    'rmse_elec_cond':  [round(ec_rmse_2,  n_round)],
                    'r2_elec_cond':    [round(ec_r2_2,    n_round)],

                    'mae_seebeck':     [round(s_mae_2,    n_round)],
                    'rmse_seebeck':    [round(s_rmse_2,   n_round)],
                    'r2_seebeck':      [round(s_r2_2,     n_round)],

                    'mae_zt':          [round(zt_mae_2,   n_round)],
                    'rmse_zt':         [round(zt_rmse_2,  n_round)],
                    'r2_zt':           [round(zt_r2_2,    n_round)],

                    'mae_k_total':     [round(k_mae_2,    n_round)],
                    'rmse_k_total':    [round(k_rmse_2,   n_round)],
                    'r2_k_total':      [round(k_r2_2,     n_round)],

                    'train_size':      [len(xTrain_temp)]
                })
            ],
            ignore_index=True
        )

        # ================= QBC SELECTION (SUM OF TWO NORMALIZED DISAGREEMENTS) =================
        pred_map = {
            "Electrical Conductivity (S/cm)": (ec_pred_val1, ec_pred_val2),
            "Seebeck Coefficient (µV/K)":     (s_pred_val1,  s_pred_val2),
            "zT":                             (zt_pred_val1, zt_pred_val2),
            "Total Thermal Conductivity (W/mK)": (k_pred_val1,  k_pred_val2),
        }

        # disagreement for target1
        p1_m1, p1_m2 = pred_map[target1]
        diff1 = np.abs(np.asarray(p1_m1) - np.asarray(p1_m2)).astype(np.float32)
        diff1_norm = minMaxStandarize(diff1)

        if target2 is None:
            # single-objective (still normalized so scale is [0,1])
            combined_score = diff1_norm
        else:
            # disagreement for target2
            p2_m1, p2_m2 = pred_map[target2]
            diff2 = np.abs(np.asarray(p2_m1) - np.asarray(p2_m2)).astype(np.float32)
            diff2_norm = minMaxStandarize(diff2)

            # sum on same scale
            combined_score = (w1 * diff1_norm) + (w2 * diff2_norm)

        # align with validation indices
        yVal_temp_diff = pd.Series(combined_score, index=yVal_temp.index)

        # pick intStepSize highest-score points
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # ---- STOP IF FULL TRAINVAL USED ----
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        # ---- UPDATE TRAIN SET ----
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        # ensure uniqueness (optional but usually good)
        lstTrainIndices = list(dict.fromkeys(lstTrainIndices))
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n------------------------------------------------------------')

    # ================= FINAL SAVE =================
    dfGrowing_QBC_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False
    )
    dfGrowing_QBC_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False
    )

    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    dfTrainIndices.to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False
    )

    return




def grow_3QBC(model1, model2,
             xTrainVal, yTrainVal,
             xTest, yTest,
             target1,
             target2=None,
             target3=None,              # <-- NEW: third disagreement target (optional)
             w1=1.0, w2=1.0, w3=1.0,     # <-- NEW: weights for sum (optional)
             strSaveDir='',
             strSaveName='growing_QBC',
             strModel1Name='rf',
             strModel2Name='xgb',
             fltStepFrac=0.01,
             intRandomSeed=0,
             n_round=6,
             n_train_ratio=4):
    """
    QBC active learning growth for thermoelectric outcomes.

    Targets (columns of yTrainVal / yTest):
        - 'Electrical Conductivity (S/cm)'
        - 'Seebeck Coefficient (µV/K)'
        - 'zT'
        - 'Total Thermal Conductivity (W/mK)'

    Disagreement score used to acquire new points:
        - If only target1:
              score = minMaxStandarize(|pred1(target1) - pred2(target1)|)
        - If target1 + target2:
              score = w1*minMaxStandarize(|pred1(t1)-pred2(t1)|)
                    + w2*minMaxStandarize(|pred1(t2)-pred2(t2)|)
        - If target1 + target2 + target3:
              score = w1*minMaxStandarize(|pred1(t1)-pred2(t1)|)
                    + w2*minMaxStandarize(|pred1(t2)-pred2(t2)|)
                    + w3*minMaxStandarize(|pred1(t3)-pred2(t3)|)

    Notes:
        - This function assumes you already have:
            * splitTrainVal(X, Y, lstTrainIndices) -> (X_train, X_val, Y_train, Y_val)
            * get_scores(model, X_train, y_train, X_test, y_test, X_val=...) ->
                  (y_pred_val, mae, rmse, r2)
    """

    # ---- TARGET COLUMNS ----
    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]

    # safety checks
    for col in target_cols:
        assert col in yTrainVal.columns, f"Column '{col}' missing from yTrainVal"
        assert col in yTest.columns,     f"Column '{col}' missing from yTest"

    def _check_target(name, val):
        if val is None:
            return
        if val not in target_cols:
            raise ValueError(
                f"{name} must be one of: "
                "'Electrical Conductivity (S/cm)', "
                "'Seebeck Coefficient (µV/K)', "
                "'zT', "
                "'Total Thermal Conductivity (W/mK)'"
            )

    _check_target("target1", target1)
    _check_target("target2", target2)
    _check_target("target3", target3)

    # normalize save path
    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    dfGrowing_QBC_model1 = pd.DataFrame()
    dfGrowing_QBC_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize      = int(len(xTrainVal) * fltStepFrac)

    # guard against tiny fractions
    intTempTrainSize = max(1, intTempTrainSize)
    intStepSize      = max(1, intStepSize)

    # initial random training indices
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    # fixed test targets
    yEC_test   = yTest["Electrical Conductivity (S/cm)"]
    yS_test    = yTest["Seebeck Coefficient (µV/K)"]
    yZT_test   = yTest["zT"]
    yKtot_test = yTest["Total Thermal Conductivity (W/mK)"]

    # ========================= MAIN LOOP =========================
    while intTempTrainSize < (len(xTrainVal) + 1):

        timeStart = time.time()

        # work only with target columns
        yTrainVal_targets = yTrainVal[target_cols]

        # split into train / val using your splitTrainVal (index-respecting)
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal, yTrainVal_targets, lstTrainIndices
        )

        print('\n------------------------------------------------------------')
        print('Training set size:', len(xTrain_temp))
        print('Training %:       ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # unpack train targets
        yEC_train   = yTrain_temp["Electrical Conductivity (S/cm)"]
        yS_train    = yTrain_temp["Seebeck Coefficient (µV/K)"]
        yZT_train   = yTrain_temp["zT"]
        yKtot_train = yTrain_temp["Total Thermal Conductivity (W/mK)"]

        # ================= MODEL 1 SCORES =================
        print("---- MODEL1: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(
            model1, xTrain_temp, yEC_train, xTest, yEC_test, X_val=xVal_temp
        )

        print("---- MODEL1: SEEBECK COEFFICIENT ----")
        s_pred_val1, s_mae_1, s_rmse_1, s_r2_1 = get_scores(
            model1, xTrain_temp, yS_train, xTest, yS_test, X_val=xVal_temp
        )

        print("---- MODEL1: zT ----")
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(
            model1, xTrain_temp, yZT_train, xTest, yZT_test, X_val=xVal_temp
        )

        print("---- MODEL1: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val1, k_mae_1, k_rmse_1, k_r2_1 = get_scores(
            model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp
        )

        # ================= MODEL 2 SCORES =================
        print("---- MODEL2: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(
            model2, xTrain_temp, yEC_train, xTest, yEC_test, X_val=xVal_temp
        )

        print("---- MODEL2: SEEBECK COEFFICIENT ----")
        s_pred_val2, s_mae_2, s_rmse_2, s_r2_2 = get_scores(
            model2, xTrain_temp, yS_train, xTest, yS_test, X_val=xVal_temp
        )

        print("---- MODEL2: zT ----")
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(
            model2, xTrain_temp, yZT_train, xTest, yZT_test, X_val=xVal_temp
        )

        print("---- MODEL2: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val2, k_mae_2, k_rmse_2, k_r2_2 = get_scores(
            model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp
        )

        # ================= STORE RESULTS =================
        dfGrowing_QBC_model1 = pd.concat(
            [
                dfGrowing_QBC_model1,
                pd.DataFrame({
                    'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                    'mae_elec_cond':   [round(ec_mae_1,   n_round)],
                    'rmse_elec_cond':  [round(ec_rmse_1,  n_round)],
                    'r2_elec_cond':    [round(ec_r2_1,    n_round)],

                    'mae_seebeck':     [round(s_mae_1,    n_round)],
                    'rmse_seebeck':    [round(s_rmse_1,   n_round)],
                    'r2_seebeck':      [round(s_r2_1,     n_round)],

                    'mae_zt':          [round(zt_mae_1,   n_round)],
                    'rmse_zt':         [round(zt_rmse_1,  n_round)],
                    'r2_zt':           [round(zt_r2_1,    n_round)],

                    'mae_k_total':     [round(k_mae_1,    n_round)],
                    'rmse_k_total':    [round(k_rmse_1,   n_round)],
                    'r2_k_total':      [round(k_r2_1,     n_round)],

                    'train_size':      [len(xTrain_temp)]
                })
            ],
            ignore_index=True
        )

        dfGrowing_QBC_model2 = pd.concat(
            [
                dfGrowing_QBC_model2,
                pd.DataFrame({
                    'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                    'mae_elec_cond':   [round(ec_mae_2,   n_round)],
                    'rmse_elec_cond':  [round(ec_rmse_2,  n_round)],
                    'r2_elec_cond':    [round(ec_r2_2,    n_round)],

                    'mae_seebeck':     [round(s_mae_2,    n_round)],
                    'rmse_seebeck':    [round(s_rmse_2,   n_round)],
                    'r2_seebeck':      [round(s_r2_2,     n_round)],

                    'mae_zt':          [round(zt_mae_2,   n_round)],
                    'rmse_zt':         [round(zt_rmse_2,  n_round)],
                    'r2_zt':           [round(zt_r2_2,    n_round)],

                    'mae_k_total':     [round(k_mae_2,    n_round)],
                    'rmse_k_total':    [round(k_rmse_2,   n_round)],
                    'r2_k_total':      [round(k_r2_2,     n_round)],

                    'train_size':      [len(xTrain_temp)]
                })
            ],
            ignore_index=True
        )

        # ================= QBC SELECTION (SUM OF UP TO THREE NORMALIZED DISAGREEMENTS) =================
        pred_map = {
            "Electrical Conductivity (S/cm)": (ec_pred_val1, ec_pred_val2),
            "Seebeck Coefficient (µV/K)":     (s_pred_val1,  s_pred_val2),
            "zT":                             (zt_pred_val1, zt_pred_val2),
            "Total Thermal Conductivity (W/mK)": (k_pred_val1,  k_pred_val2),
        }

        def _norm_disagreement(tname):
            """Return min-max normalized disagreement for one target (len = |val|)."""
            p_m1, p_m2 = pred_map[tname]
            diff = np.abs(np.asarray(p_m1) - np.asarray(p_m2)).astype(np.float32)
            return minMaxStandarize(diff)

        # always include target1
        combined_score = (w1 * _norm_disagreement(target1))

        # optionally add target2/3
        if target2 is not None:
            combined_score = combined_score + (w2 * _norm_disagreement(target2))
        if target3 is not None:
            combined_score = combined_score + (w3 * _norm_disagreement(target3))

        # align with validation indices
        yVal_temp_diff = pd.Series(combined_score, index=yVal_temp.index)

        # pick intStepSize highest-score points
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # ---- STOP IF FULL TRAINVAL USED ----
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        # ---- UPDATE TRAIN SET ----
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        # ensure uniqueness (optional but usually good)
        lstTrainIndices = list(dict.fromkeys(lstTrainIndices))
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n------------------------------------------------------------')

    # ================= FINAL SAVE =================
    dfGrowing_QBC_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False
    )
    dfGrowing_QBC_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False
    )

    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    dfTrainIndices.to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False
    )

    return



def grow_multiobj_all(model1, model2,
                      xTrainVal, yTrainVal,
                      xTest, yTest,
                      strSaveDir='',
                      strSaveName='growing_QBC_all',
                      strModel1Name='rf',
                      strModel2Name='xgb',
                      fltStepFrac=0.01,
                      intRandomSeed=0,
                      n_round=6,
                      n_train_ratio=4,
                      weights=None):
    """
    QBC active learning growth for thermoelectric outcomes using ALL 4 targets.

    Targets (columns of yTrainVal / yTest):
        - 'Electrical Conductivity (S/cm)'
        - 'Seebeck Coefficient (µV/K)'
        - 'zT'
        - 'Total Thermal Conductivity (W/mK)'

    Acquisition score used to acquire new points:
        score = sum_k w_k * minMaxStandarize(|pred1(target_k) - pred2(target_k)|)
    where each disagreement is min-max normalized to [0,1] on the CURRENT validation pool,
    ensuring all outcomes are on the same scale before summing.

    Parameters
    ----------
    weights : dict or None
        Optional weights per target. Example:
            {
              "zT": 1.0,
              "Seebeck Coefficient (µV/K)": 1.0,
              "Electrical Conductivity (S/cm)": 0.5,
              "Total Thermal Conductivity (W/mK)": 0.5
            }
        If None, all weights = 1.0.

    Notes
    -----
    This function assumes you already have:
        * splitTrainVal(X, Y, lstTrainIndices) -> (X_train, X_val, Y_train, Y_val)
        * get_scores(model, X_train, y_train, X_test, y_test, X_val=...) ->
              (y_pred_val, mae, rmse, r2)
    """

    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]

    # --- safety checks ---
    for col in target_cols:
        assert col in yTrainVal.columns, f"Column '{col}' missing from yTrainVal"
        assert col in yTest.columns,     f"Column '{col}' missing from yTest"

    # --- weights ---
    if weights is None:
        weights = {t: 1.0 for t in target_cols}
    else:
        # fill missing keys with 1.0, ignore extras
        weights = {t: float(weights.get(t, 1.0)) for t in target_cols}

    # normalize save path
    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    dfGrowing_QBC_model1 = pd.DataFrame()
    dfGrowing_QBC_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize      = int(len(xTrainVal) * fltStepFrac)

    # guard against tiny fractions
    intTempTrainSize = max(1, intTempTrainSize)
    intStepSize      = max(1, intStepSize)

    # initial random training indices
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    # fixed test targets
    yEC_test   = yTest["Electrical Conductivity (S/cm)"]
    yS_test    = yTest["Seebeck Coefficient (µV/K)"]
    yZT_test   = yTest["zT"]
    yKtot_test = yTest["Total Thermal Conductivity (W/mK)"]

    # ========================= MAIN LOOP =========================
    while intTempTrainSize < (len(xTrainVal) + 1):

        timeStart = time.time()

        # work only with target columns
        yTrainVal_targets = yTrainVal[target_cols]

        # split into train / val using your splitTrainVal (index-respecting)
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal, yTrainVal_targets, lstTrainIndices
        )

        print('\n------------------------------------------------------------')
        print('Training set size:', len(xTrain_temp))
        print('Training %:       ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # unpack train targets
        yEC_train   = yTrain_temp["Electrical Conductivity (S/cm)"]
        yS_train    = yTrain_temp["Seebeck Coefficient (µV/K)"]
        yZT_train   = yTrain_temp["zT"]
        yKtot_train = yTrain_temp["Total Thermal Conductivity (W/mK)"]

        # ================= MODEL 1 SCORES =================
        print("---- MODEL1: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(
            model1, xTrain_temp, yEC_train, xTest, yEC_test, X_val=xVal_temp
        )

        print("---- MODEL1: SEEBECK COEFFICIENT ----")
        s_pred_val1, s_mae_1, s_rmse_1, s_r2_1 = get_scores(
            model1, xTrain_temp, yS_train, xTest, yS_test, X_val=xVal_temp
        )

        print("---- MODEL1: zT ----")
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(
            model1, xTrain_temp, yZT_train, xTest, yZT_test, X_val=xVal_temp
        )

        print("---- MODEL1: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val1, k_mae_1, k_rmse_1, k_r2_1 = get_scores(
            model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp
        )

        # ================= MODEL 2 SCORES =================
        print("---- MODEL2: ELECTRICAL CONDUCTIVITY ----")
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(
            model2, xTrain_temp, yEC_train, xTest, yEC_test, X_val=xVal_temp
        )

        print("---- MODEL2: SEEBECK COEFFICIENT ----")
        s_pred_val2, s_mae_2, s_rmse_2, s_r2_2 = get_scores(
            model2, xTrain_temp, yS_train, xTest, yS_test, X_val=xVal_temp
        )

        print("---- MODEL2: zT ----")
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(
            model2, xTrain_temp, yZT_train, xTest, yZT_test, X_val=xVal_temp
        )

        print("---- MODEL2: TOTAL THERMAL CONDUCTIVITY ----")
        k_pred_val2, k_mae_2, k_rmse_2, k_r2_2 = get_scores(
            model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp
        )

        # ================= STORE RESULTS =================
        dfGrowing_QBC_model1 = pd.concat(
            [dfGrowing_QBC_model1,
             pd.DataFrame({
                 'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                 'mae_elec_cond':   [round(ec_mae_1,   n_round)],
                 'rmse_elec_cond':  [round(ec_rmse_1,  n_round)],
                 'r2_elec_cond':    [round(ec_r2_1,    n_round)],

                 'mae_seebeck':     [round(s_mae_1,    n_round)],
                 'rmse_seebeck':    [round(s_rmse_1,   n_round)],
                 'r2_seebeck':      [round(s_r2_1,     n_round)],

                 'mae_zt':          [round(zt_mae_1,   n_round)],
                 'rmse_zt':         [round(zt_rmse_1,  n_round)],
                 'r2_zt':           [round(zt_r2_1,    n_round)],

                 'mae_k_total':     [round(k_mae_1,    n_round)],
                 'rmse_k_total':    [round(k_rmse_1,   n_round)],
                 'r2_k_total':      [round(k_r2_1,     n_round)],

                 'train_size':      [len(xTrain_temp)]
             })],
            ignore_index=True
        )

        dfGrowing_QBC_model2 = pd.concat(
            [dfGrowing_QBC_model2,
             pd.DataFrame({
                 'train_ratio':     [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],

                 'mae_elec_cond':   [round(ec_mae_2,   n_round)],
                 'rmse_elec_cond':  [round(ec_rmse_2,  n_round)],
                 'r2_elec_cond':    [round(ec_r2_2,    n_round)],

                 'mae_seebeck':     [round(s_mae_2,    n_round)],
                 'rmse_seebeck':    [round(s_rmse_2,   n_round)],
                 'r2_seebeck':      [round(s_r2_2,     n_round)],

                 'mae_zt':          [round(zt_mae_2,   n_round)],
                 'rmse_zt':         [round(zt_rmse_2,  n_round)],
                 'r2_zt':           [round(zt_r2_2,    n_round)],

                 'mae_k_total':     [round(k_mae_2,    n_round)],
                 'rmse_k_total':    [round(k_rmse_2,   n_round)],
                 'r2_k_total':      [round(k_r2_2,     n_round)],

                 'train_size':      [len(xTrain_temp)]
             })],
            ignore_index=True
        )

        # ================= QBC SELECTION (ALL 4 NORMALIZED DISAGREEMENTS) =================
        pred_map = {
            "Electrical Conductivity (S/cm)": (ec_pred_val1, ec_pred_val2),
            "Seebeck Coefficient (µV/K)":     (s_pred_val1,  s_pred_val2),
            "zT":                             (zt_pred_val1, zt_pred_val2),
            "Total Thermal Conductivity (W/mK)": (k_pred_val1,  k_pred_val2),
        }

        def _norm_disagreement(tname):
            p_m1, p_m2 = pred_map[tname]
            diff = np.abs(np.asarray(p_m1) - np.asarray(p_m2)).astype(np.float32)
            return minMaxStandarize(diff)

        combined_score = np.zeros(len(xVal_temp), dtype=np.float32)
        for t in target_cols:
            combined_score += float(weights[t]) * _norm_disagreement(t)

        yVal_temp_diff = pd.Series(combined_score, index=yVal_temp.index)
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # ---- STOP IF FULL TRAINVAL USED ----
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        # ---- UPDATE TRAIN SET ----
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        lstTrainIndices = list(dict.fromkeys(lstTrainIndices))  # unique, preserve order
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n------------------------------------------------------------')

    # ================= FINAL SAVE =================
    dfGrowing_QBC_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False
    )
    dfGrowing_QBC_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False
    )

    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    dfTrainIndices.to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False
    )

    return






# =========================
# NSGA-II components (works for M objectives)
# =========================
def _fast_non_dominated_sort(F):
    """
    F: (pop, M), higher is better for all objectives.
    Returns: fronts (list of lists), rank (array)
    """
    pop = F.shape[0]
    S = [[] for _ in range(pop)]
    n = np.zeros(pop, dtype=int)
    rank = np.zeros(pop, dtype=int)
    fronts = [[]]

    for p in range(pop):
        for q in range(pop):
            if p == q:
                continue
            # p dominates q: >= in all and > in at least one
            if np.all(F[p] >= F[q]) and np.any(F[p] > F[q]):
                S[p].append(q)
            elif np.all(F[q] >= F[p]) and np.any(F[q] > F[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, rank


def _crowding_distance(F, front):
    """
    F: (pop, M), front: list of indices
    Returns: dict idx->crowding (higher is better)
    """
    if len(front) == 0:
        return {}
    if len(front) <= 2:
        return {i: np.inf for i in front}

    crowd = {i: 0.0 for i in front}
    M = F.shape[1]

    for m in range(M):
        vals = np.array([F[i, m] for i in front], dtype=float)
        order = np.argsort(vals)
        sorted_idx = [front[j] for j in order]

        crowd[sorted_idx[0]] = np.inf
        crowd[sorted_idx[-1]] = np.inf

        vmin = vals[order[0]]
        vmax = vals[order[-1]]
        denom = (vmax - vmin) if (vmax > vmin) else 1.0

        for k in range(1, len(front) - 1):
            i_mid = sorted_idx[k]
            i_prev = sorted_idx[k - 1]
            i_next = sorted_idx[k + 1]
            crowd[i_mid] += (F[i_next, m] - F[i_prev, m]) / denom

    return crowd


def _tournament_select(pop_indices, rank, crowd, rng, k=2):
    """Binary tournament using (lower rank better) then (higher crowding better)."""
    best = None
    for _ in range(k):
        i = int(rng.choice(pop_indices))
        if best is None:
            best = i
            continue
        if rank[i] < rank[best]:
            best = i
        elif rank[i] == rank[best]:
            if crowd.get(i, 0.0) > crowd.get(best, 0.0):
                best = i
    return best


def _make_unique(chrom, pool_size, rng):
    """Repair duplicates by resampling."""
    chrom = np.array(chrom, dtype=int)
    used = set()
    for t in range(len(chrom)):
        if int(chrom[t]) in used:
            while True:
                r = int(rng.integers(0, pool_size))
                if r not in used:
                    chrom[t] = r
                    break
        used.add(int(chrom[t]))
    return chrom


def _ordered_crossover(a, b, rng):
    """Safe order crossover (OX) for set/list chromosomes."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    B = a.size
    if B < 2:
        return a.copy(), b.copy()

    i, j = sorted(rng.choice(np.arange(B), size=2, replace=False))
    if i == j:
        j = min(B, i + 1)

    child1 = np.full(B, -1, dtype=int)
    child2 = np.full(B, -1, dtype=int)

    child1[i:j] = a[i:j]
    child2[i:j] = b[i:j]

    def fill(child, parent):
        taken = set(child[child != -1].tolist())
        remaining = [g for g in parent.tolist() if g not in taken]
        empty_pos = np.where(child == -1)[0]
        m = min(len(empty_pos), len(remaining))
        child[empty_pos[:m]] = remaining[:m]
        if np.any(child == -1):
            left = np.where(child == -1)[0]
            child[left] = rng.choice(parent, size=left.size, replace=True)
        return child

    return fill(child1, b), fill(child2, a)


def _mutate_swap(chrom, pool_size, rng, p_mut=0.2):
    """Replace one gene with random pool index."""
    chrom = chrom.copy()
    if rng.random() < p_mut:
        t = int(rng.integers(0, len(chrom)))
        chrom[t] = int(rng.integers(0, pool_size))
    return chrom



def EA_select_batch_delta_diversity(
    trainingdata: pd.DataFrame,
    candidates: pd.DataFrame,
    diffs: pd.Series,                 # disagreement / "delta" aligned to candidates.index
    step_size: int,
    do_scale: bool = True,
    normalize: bool = True,
    # EA hyperparams
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
    # diversity mixing
    w_train: float = 0.5,             # weight for diversity-to-train in diversity objective
    w_within: float = 0.5,            # weight for within-batch diversity in diversity objective
):
    """
    Returns: list of candidate indices (length <= step_size)

    Objective 1 (delta/quality): mean(diffs[batch])
    Objective 2 (diversity): w_train*(1-mean cos to train) + w_within*(1-mean pairwise cos within batch)

    Uses an NSGA-II-like loop (non-dominated sorting + crowding + tournament).
    """

    rng = np.random.default_rng(random_state)

    X_train = trainingdata.values.astype(np.float32, copy=False)
    X_pool  = candidates.values.astype(np.float32, copy=False)

    # Fit scaler on stack (train + pool) to define a consistent geometry
    if do_scale:
        scaler = StandardScaler()
        X_all0 = np.vstack([X_train, X_pool])
        scaler.fit(X_all0)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_pool  = scaler.transform(X_pool).astype(np.float32, copy=False)

    if normalize:
        X_train = sk_normalize(X_train, axis=1)
        X_pool  = sk_normalize(X_pool, axis=1)

    n_train = X_train.shape[0]
    n_pool  = X_pool.shape[0]
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    # Quality vector aligned
    if isinstance(diffs, pd.Series):
        q = diffs.reindex(candidates.index).values.astype(np.float32)
    else:
        q = np.asarray(diffs, dtype=np.float32).reshape(-1)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    # Precompute similarities needed for fitness:
    # pool-to-train mean cosine similarity per pool point
    if n_train > 0:
        sim_pool_train = (X_pool @ X_train.T).mean(axis=1).astype(np.float32)  # (n_pool,)
    else:
        sim_pool_train = np.zeros(n_pool, dtype=np.float32)

    # pool-to-pool cosine Gram (for within-batch diversity)
    # NOTE: n_pool up to ~10k => 10k^2 float32 ~ 400MB; OK in your stated range, but still heavy.
    G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

    def fitness(chrom):
        # chrom: indices in [0..n_pool-1], length B
        idx = np.asarray(chrom, dtype=int)

        # Objective 1: quality (delta)
        qual = float(np.mean(q[idx]))

        # Diversity-to-train: higher is better
        div_train = float(np.mean(1.0 - sim_pool_train[idx]))

        # Within-batch diversity: 1 - mean pairwise cosine similarity
        if len(idx) <= 1:
            div_within = 1.0
        else:
            S = G_pp[np.ix_(idx, idx)]
            # mean of off-diagonal
            off = S[~np.eye(len(idx), dtype=bool)]
            div_within = float(1.0 - np.mean(off))

        div = w_train * div_train + w_within * div_within
        return qual, div

    # ---------- init population ----------
    pop = []
    for _ in range(pop_size):
        chrom = rng.choice(n_pool, size=B, replace=False)
        pop.append(chrom.astype(int))

    # ---------- evolution loop ----------
    for _ in range(n_gen):
        # Evaluate
        F = np.array([fitness(ind) for ind in pop], dtype=float)

        # Non-dominated sorting + crowding
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        # Create offspring
        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            # ✅ repair immediately after crossover
            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            # ✅ repair again after mutation
            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)


            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # Combine and select next gen via NSGA-II elitism
        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, rank2 = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                # fill remaining by crowding distance
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break

        pop = new_pop

    # ---------- pick final batch ----------
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]

    # choose one solution from Pareto front: max of normalized (qual + div)/2
    f0 = F_final[front0]
    # normalize each objective on front
    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = 0.5 * f0n[:, 0] + 0.5 * f0n[:, 1]
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    best_indices = candidates.index[np.asarray(best_local, dtype=int)].tolist()
    return best_indices


# ============================================================
# Active learning loop using EA (delta + diversity)
# You can drop this into your grow_QVendi pipeline by replacing
# the batch selection call.
# ============================================================

def grow_EA_DeltaDiversity(
    model1, model2,
    xTrainVal, yTrainVal,
    xTest, yTest,
    target,
    strSaveDir='',
    strSaveName='growing_EA',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    # geometry
    do_scale_global=True,
    normalize_global=True,
    # EA params
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    # diversity mix
    w_train=0.5,
    w_within=0.5,
):
    """
    Same structure as your grow_QVendi, but batch selection is done by an EA that
    optimizes TWO objectives per iteration:
      - delta/quality (committee disagreement on target)
      - diversity (to training + within-batch)

    Requires your existing:
      - splitTrainVal(...)
      - get_scores(...)
      - n_round, n_train_ratio
    """

    import random, time
    import numpy as np
    import pandas as pd

    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]
    if target not in target_cols:
        raise ValueError("target must be one of the target_cols")

    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    lstTrainIndices = np.random.choice(xTrainVal.index, init_size, replace=False).tolist()

    yEC_test, yS_test, yZT_test, yKtot_test = (yTest[c] for c in target_cols)

    while True:
        t0 = time.time()

        yTV = yTrainVal[target_cols]
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTV, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.1%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        yEC_train, yS_train, yZT_train, yKtot_train = (yTrain_temp[c] for c in target_cols)

        # model 1
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(model1, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val1,  s_mae_1,  s_rmse_1,  s_r2_1  = get_scores(model1, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(model1, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val1,  k_mae_1,  k_rmse_1,  k_r2_1  = get_scores(model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # model 2
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(model2, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val2,  s_mae_2,  s_rmse_2,  s_r2_2  = get_scores(model2, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(model2, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val2,  k_mae_2,  k_rmse_2,  k_r2_2  = get_scores(model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # log (same schema you used)
        df1 = pd.concat([df1, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_1, n_round)], "rmse_elec_cond":[round(ec_rmse_1, n_round)], "r2_elec_cond":[round(ec_r2_1, n_round)],
            "mae_seebeck":[round(s_mae_1, n_round)],    "rmse_seebeck":[round(s_rmse_1, n_round)],    "r2_seebeck":[round(s_r2_1, n_round)],
            "mae_zt":[round(zt_mae_1, n_round)],        "rmse_zt":[round(zt_rmse_1, n_round)],        "r2_zt":[round(zt_r2_1, n_round)],
            "mae_k_total":[round(k_mae_1, n_round)],    "rmse_k_total":[round(k_rmse_1, n_round)],    "r2_k_total":[round(k_r2_1, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        df2 = pd.concat([df2, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_2, n_round)], "rmse_elec_cond":[round(ec_rmse_2, n_round)], "r2_elec_cond":[round(ec_r2_2, n_round)],
            "mae_seebeck":[round(s_mae_2, n_round)],    "rmse_seebeck":[round(s_rmse_2, n_round)],    "r2_seebeck":[round(s_r2_2, n_round)],
            "mae_zt":[round(zt_mae_2, n_round)],        "rmse_zt":[round(zt_rmse_2, n_round)],        "r2_zt":[round(zt_r2_2, n_round)],
            "mae_k_total":[round(k_mae_2, n_round)],    "rmse_k_total":[round(k_rmse_2, n_round)],    "r2_k_total":[round(k_r2_2, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        if len(xVal_temp) == 0:
            print("No validation candidates left. Stopping.")
            break

        # disagreement (delta/quality) on chosen target
        if target == target_cols[0]:
            val_pred1, val_pred2 = ec_pred_val1, ec_pred_val2
        elif target == target_cols[1]:
            val_pred1, val_pred2 = s_pred_val1, s_pred_val2
        elif target == target_cols[2]:
            val_pred1, val_pred2 = zt_pred_val1, zt_pred_val2
        else:
            val_pred1, val_pred2 = k_pred_val1, k_pred_val2

        diffs = pd.Series(np.abs(val_pred1 - val_pred2), index=yVal_temp.index)

        # EA selection (delta + diversity)

        print('BEFORE EA ')
        to_add = EA_select_batch_delta_diversity(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            diffs=diffs,
            step_size=min(step_size, len(xVal_temp)),
            do_scale=do_scale_global,
            normalize=normalize_global,
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
            w_train=w_train,
            w_within=w_within
        )
        print('AFTER EA')

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices = lstTrainIndices + to_add

        print("Time to run loop:", round(time.time() - t0, 2), "seconds", flush=True)

    # save
    df1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False
    )
    return





# =========================
# 3-objective EA batch selection:
#   (quality1, quality2, diversity)
# =========================
def EA_select_batch_twoQuality_plus_diversity(
    trainingdata: pd.DataFrame,
    candidates: pd.DataFrame,
    diffs_df: pd.DataFrame,          # columns: ['q1','q2'] aligned to candidates.index
    step_size: int,
    do_scale: bool = True,
    normalize: bool = True,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
    # diversity mixing
    w_train: float = 0.5,
    w_within: float = 0.5,
):
    """
    Objectives (maximize):
      1) mean q1 over batch
      2) mean q2 over batch
      3) diversity = w_train*(1-mean cos to train) + w_within*(1-mean pairwise cos within batch)

    Returns: list of candidates.index of selected batch
    """
    rng = np.random.default_rng(random_state)

    X_train = trainingdata.values.astype(np.float32, copy=False)
    X_pool  = candidates.values.astype(np.float32, copy=False)

    # Fit scaler on stacked train+pool
    if do_scale:
        scaler = StandardScaler()
        X_all0 = np.vstack([X_train, X_pool])
        scaler.fit(X_all0)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_pool  = scaler.transform(X_pool).astype(np.float32, copy=False)

    if normalize:
        X_train = sk_normalize(X_train, axis=1)
        X_pool  = sk_normalize(X_pool, axis=1)

    n_train = X_train.shape[0]
    n_pool  = X_pool.shape[0]
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    # quality vectors aligned
    diffs_df = diffs_df.reindex(candidates.index)
    q1 = np.nan_to_num(diffs_df.iloc[:, 0].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q2 = np.nan_to_num(diffs_df.iloc[:, 1].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # precompute
    if n_train > 0:
        sim_pool_train = (X_pool @ X_train.T).mean(axis=1).astype(np.float32)
    else:
        sim_pool_train = np.zeros(n_pool, dtype=np.float32)

    G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)

        qual1 = float(np.mean(q1[idx]))
        qual2 = float(np.mean(q2[idx]))

        div_train = float(np.mean(1.0 - sim_pool_train[idx]))
        if len(idx) <= 1:
            div_within = 1.0
        else:
            S = G_pp[np.ix_(idx, idx)]
            off = S[~np.eye(len(idx), dtype=bool)]
            div_within = float(1.0 - np.mean(off))

        div = w_train * div_train + w_within * div_within
        return qual1, qual2, div  # 3 objectives

    # init pop
    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,3)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # ===== pick a single solution from Pareto front =====
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,3)

    # normalize each objective on front and pick max average
    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)  # equal weights for (q1,q2,div)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()


# =========================
# Grow loop (TE target_cols) using 3-objective EA
# =========================
def grow_EA_3Obj_TwoTargets(
    model1, model2,
    xTrainVal, yTrainVal,
    xTest, yTest,
    target1: str,
    target2: str,
    strSaveDir='',
    strSaveName='growing_EA_3obj',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    w_train=0.5,
    w_within=0.5,
):
    """
    3-objective NSGA-II:
      maximize mean(diff(target1)), mean(diff(target2)), diversity
    where diffs are min-max standardized per-iteration on validation predictions.

    Requires your existing:
      - splitTrainVal(...)
      - get_scores(...)
      - n_round, n_train_ratio
    """

    import random, time
    import numpy as np
    import pandas as pd

    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]
    if target1 not in target_cols or target2 not in target_cols:
        raise ValueError(f"target1/target2 must be in: {target_cols}")

    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    lstTrainIndices = np.random.choice(xTrainVal.index, init_size, replace=False).tolist()

    # fixed test targets
    yEC_test   = yTest[target_cols[0]]
    yS_test    = yTest[target_cols[1]]
    yZT_test   = yTest[target_cols[2]]
    yKtot_test = yTest[target_cols[3]]

    while True:
        t0 = time.time()

        yTV = yTrainVal[target_cols]
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTV, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.1%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        # unpack train targets
        yEC_train   = yTrain_temp[target_cols[0]]
        yS_train    = yTrain_temp[target_cols[1]]
        yZT_train   = yTrain_temp[target_cols[2]]
        yKtot_train = yTrain_temp[target_cols[3]]

        # model 1
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(model1, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val1,  s_mae_1,  s_rmse_1,  s_r2_1  = get_scores(model1, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(model1, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val1,  k_mae_1,  k_rmse_1,  k_r2_1  = get_scores(model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # model 2
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(model2, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val2,  s_mae_2,  s_rmse_2,  s_r2_2  = get_scores(model2, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(model2, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val2,  k_mae_2,  k_rmse_2,  k_r2_2  = get_scores(model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # store results (same schema as before)
        df1 = pd.concat([df1, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_1, n_round)], "rmse_elec_cond":[round(ec_rmse_1, n_round)], "r2_elec_cond":[round(ec_r2_1, n_round)],
            "mae_seebeck":[round(s_mae_1, n_round)],    "rmse_seebeck":[round(s_rmse_1, n_round)],    "r2_seebeck":[round(s_r2_1, n_round)],
            "mae_zt":[round(zt_mae_1, n_round)],        "rmse_zt":[round(zt_rmse_1, n_round)],        "r2_zt":[round(zt_r2_1, n_round)],
            "mae_k_total":[round(k_mae_1, n_round)],    "rmse_k_total":[round(k_rmse_1, n_round)],    "r2_k_total":[round(k_r2_1, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        df2 = pd.concat([df2, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_2, n_round)], "rmse_elec_cond":[round(ec_rmse_2, n_round)], "r2_elec_cond":[round(ec_r2_2, n_round)],
            "mae_seebeck":[round(s_mae_2, n_round)],    "rmse_seebeck":[round(s_rmse_2, n_round)],    "r2_seebeck":[round(s_r2_2, n_round)],
            "mae_zt":[round(zt_mae_2, n_round)],        "rmse_zt":[round(zt_rmse_2, n_round)],        "r2_zt":[round(zt_r2_2, n_round)],
            "mae_k_total":[round(k_mae_2, n_round)],    "rmse_k_total":[round(k_rmse_2, n_round)],    "r2_k_total":[round(k_r2_2, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        if len(xVal_temp) == 0:
            print("No validation candidates left. Stopping.")
            break

        # -------- compute per-target diffs on validation --------
        pred_map_1 = {
            target_cols[0]: ec_pred_val1,
            target_cols[1]: s_pred_val1,
            target_cols[2]: zt_pred_val1,
            target_cols[3]: k_pred_val1,
        }
        pred_map_2 = {
            target_cols[0]: ec_pred_val2,
            target_cols[1]: s_pred_val2,
            target_cols[2]: zt_pred_val2,
            target_cols[3]: k_pred_val2,
        }

        d1 = np.abs(pred_map_1[target1] - pred_map_2[target1])
        d2 = np.abs(pred_map_1[target2] - pred_map_2[target2])

        d1s = minMaxStandarize(d1)
        d2s = minMaxStandarize(d2)

        diffs_df = pd.DataFrame(
            {"q1": d1s, "q2": d2s},
            index=yVal_temp.index
        )

        # -------- EA selection (3-objective) --------
        to_add = EA_select_batch_twoQuality_plus_diversity(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(step_size, len(xVal_temp)),
            do_scale=do_scale_global,
            normalize=normalize_global,
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
            w_train=w_train,
            w_within=w_within
        )

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices = lstTrainIndices + to_add
        print("Time to run loop:", round(time.time() - t0, 2), "seconds", flush=True)

    # save
    df1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False
    )
    return




# =========================
# 4-objective EA batch selection:
#   (quality1, quality2, quality3, diversity)
# =========================
def EA_select_batch_threeQuality_plus_diversity(
    trainingdata: pd.DataFrame,
    candidates: pd.DataFrame,
    diffs_df: pd.DataFrame,          # columns: ['q1','q2','q3'] aligned to candidates.index
    step_size: int,
    do_scale: bool = True,
    normalize: bool = True,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
    # diversity mixing
    w_train: float = 0.5,
    w_within: float = 0.5,
):
    """
    Objectives (maximize):
      1) mean q1 over batch
      2) mean q2 over batch
      3) mean q3 over batch
      4) diversity = w_train*(1-mean cos to train) + w_within*(1-mean pairwise cos within batch)

    Returns: list of candidates.index of selected batch
    """
    rng = np.random.default_rng(random_state)

    X_train = trainingdata.values.astype(np.float32, copy=False)
    X_pool  = candidates.values.astype(np.float32, copy=False)

    # Fit scaler on stacked train+pool
    if do_scale:
        scaler = StandardScaler()
        X_all0 = np.vstack([X_train, X_pool])
        scaler.fit(X_all0)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_pool  = scaler.transform(X_pool).astype(np.float32, copy=False)

    if normalize:
        X_train = sk_normalize(X_train, axis=1)
        X_pool  = sk_normalize(X_pool, axis=1)

    n_train = X_train.shape[0]
    n_pool  = X_pool.shape[0]
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    # quality vectors aligned
    diffs_df = diffs_df.reindex(candidates.index)
    if diffs_df.shape[1] < 3:
        raise ValueError("diffs_df must have at least 3 columns: ['q1','q2','q3']")

    q1 = np.nan_to_num(diffs_df.iloc[:, 0].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q2 = np.nan_to_num(diffs_df.iloc[:, 1].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q3 = np.nan_to_num(diffs_df.iloc[:, 2].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # precompute
    if n_train > 0:
        sim_pool_train = (X_pool @ X_train.T).mean(axis=1).astype(np.float32)
    else:
        sim_pool_train = np.zeros(n_pool, dtype=np.float32)

    G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)

        qual1 = float(np.mean(q1[idx]))
        qual2 = float(np.mean(q2[idx]))
        qual3 = float(np.mean(q3[idx]))

        div_train = float(np.mean(1.0 - sim_pool_train[idx]))
        if len(idx) <= 1:
            div_within = 1.0
        else:
            S = G_pp[np.ix_(idx, idx)]
            off = S[~np.eye(len(idx), dtype=bool)]
            div_within = float(1.0 - np.mean(off))

        div = w_train * div_train + w_within * div_within
        return qual1, qual2, qual3, div  # 4 objectives

    # init pop
    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,4)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # ===== pick a single solution from Pareto front =====
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,4)

    # normalize each objective on front and pick max average
    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)  # equal weights for (q1,q2,q3,div)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()


# =========================
# Grow loop (TE target_cols) using 4-objective EA:
#   (diff(target1), diff(target2), diff(target3), diversity)
# =========================
def grow_EA_4Obj_ThreeTargets(
    model1, model2,
    xTrainVal, yTrainVal,
    xTest, yTest,
    target1: str,
    target2: str,
    target3: str,
    strSaveDir='',
    strSaveName='growing_EA_4obj',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    w_train=0.5,
    w_within=0.5,
):
    """
    4-objective NSGA-II:
      maximize mean(diff(target1)), mean(diff(target2)), mean(diff(target3)), diversity
    where diffs are min-max standardized per-iteration on validation predictions.

    Requires your existing:
      - splitTrainVal(...)
      - get_scores(...)
      - n_round, n_train_ratio
      - EA helper funcs: _fast_non_dominated_sort, _crowding_distance, _tournament_select,
                        _ordered_crossover, _make_unique, _mutate_swap
    """

    import random, time
    import numpy as np
    import pandas as pd

    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]
    for t in [target1, target2, target3]:
        if t not in target_cols:
            raise ValueError(f"Targets must be in: {target_cols}")

    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    lstTrainIndices = np.random.choice(xTrainVal.index, init_size, replace=False).tolist()

    # fixed test targets
    yEC_test   = yTest[target_cols[0]]
    yS_test    = yTest[target_cols[1]]
    yZT_test   = yTest[target_cols[2]]
    yKtot_test = yTest[target_cols[3]]

    while True:
        t0 = time.time()

        yTV = yTrainVal[target_cols]
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTV, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.1%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        # unpack train targets
        yEC_train   = yTrain_temp[target_cols[0]]
        yS_train    = yTrain_temp[target_cols[1]]
        yZT_train   = yTrain_temp[target_cols[2]]
        yKtot_train = yTrain_temp[target_cols[3]]

        # model 1
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(model1, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val1,  s_mae_1,  s_rmse_1,  s_r2_1  = get_scores(model1, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(model1, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val1,  k_mae_1,  k_rmse_1,  k_r2_1  = get_scores(model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # model 2
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(model2, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val2,  s_mae_2,  s_rmse_2,  s_r2_2  = get_scores(model2, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(model2, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val2,  k_mae_2,  k_rmse_2,  k_r2_2  = get_scores(model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # store results (same schema as before)
        df1 = pd.concat([df1, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_1, n_round)], "rmse_elec_cond":[round(ec_rmse_1, n_round)], "r2_elec_cond":[round(ec_r2_1, n_round)],
            "mae_seebeck":[round(s_mae_1, n_round)],    "rmse_seebeck":[round(s_rmse_1, n_round)],    "r2_seebeck":[round(s_r2_1, n_round)],
            "mae_zt":[round(zt_mae_1, n_round)],        "rmse_zt":[round(zt_rmse_1, n_round)],        "r2_zt":[round(zt_r2_1, n_round)],
            "mae_k_total":[round(k_mae_1, n_round)],    "rmse_k_total":[round(k_rmse_1, n_round)],    "r2_k_total":[round(k_r2_1, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        df2 = pd.concat([df2, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_2, n_round)], "rmse_elec_cond":[round(ec_rmse_2, n_round)], "r2_elec_cond":[round(ec_r2_2, n_round)],
            "mae_seebeck":[round(s_mae_2, n_round)],    "rmse_seebeck":[round(s_rmse_2, n_round)],    "r2_seebeck":[round(s_r2_2, n_round)],
            "mae_zt":[round(zt_mae_2, n_round)],        "rmse_zt":[round(zt_rmse_2, n_round)],        "r2_zt":[round(zt_r2_2, n_round)],
            "mae_k_total":[round(k_mae_2, n_round)],    "rmse_k_total":[round(k_rmse_2, n_round)],    "r2_k_total":[round(k_r2_2, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        if len(xVal_temp) == 0:
            print("No validation candidates left. Stopping.")
            break

        # -------- compute per-target diffs on validation --------
        pred_map_1 = {
            target_cols[0]: ec_pred_val1,
            target_cols[1]: s_pred_val1,
            target_cols[2]: zt_pred_val1,
            target_cols[3]: k_pred_val1,
        }
        pred_map_2 = {
            target_cols[0]: ec_pred_val2,
            target_cols[1]: s_pred_val2,
            target_cols[2]: zt_pred_val2,
            target_cols[3]: k_pred_val2,
        }

        d1 = np.abs(pred_map_1[target1] - pred_map_2[target1])
        d2 = np.abs(pred_map_1[target2] - pred_map_2[target2])
        d3 = np.abs(pred_map_1[target3] - pred_map_2[target3])

        d1s = minMaxStandarize(d1)
        d2s = minMaxStandarize(d2)
        d3s = minMaxStandarize(d3)

        diffs_df = pd.DataFrame(
            {"q1": d1s, "q2": d2s, "q3": d3s},
            index=yVal_temp.index
        )

        # -------- EA selection (4-objective) --------
        to_add = EA_select_batch_threeQuality_plus_diversity(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(step_size, len(xVal_temp)),
            do_scale=do_scale_global,
            normalize=normalize_global,
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
            w_train=w_train,
            w_within=w_within
        )

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices = lstTrainIndices + to_add
        print("Time to run loop:", round(time.time() - t0, 2), "seconds", flush=True)

        # stop if we've used everything
        if len(lstTrainIndices) >= len(xTrainVal):
            print("Reached full training set. Stopping.")
            break

    # save
    df1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False
    )
    return



# =========================
# 2-objective EA batch selection:
#   (quality1, quality2)
# NO diversity
# =========================
def EA_select_batch_twoQuality_only(
    candidates: pd.DataFrame,
    diffs_df: pd.DataFrame,          # columns ['q1','q2'] aligned to candidates.index
    step_size: int,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
):
    """
    Objectives (maximize):
      1) mean q1 over batch
      2) mean q2 over batch

    Returns: list of candidates.index (length <= step_size)
    """
    rng = np.random.default_rng(random_state)

    n_pool = len(candidates)
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    diffs_df = diffs_df.reindex(candidates.index)
    q1 = np.nan_to_num(diffs_df["q1"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q2 = np.nan_to_num(diffs_df["q2"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)
        return float(np.mean(q1[idx])), float(np.mean(q2[idx]))  # 2 objectives

    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,2)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # pick one solution from Pareto front
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,2)

    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()


# =========================
# Grow loop using 2-objective EA:
#   objectives = normalized abs QBC diffs for (target1, target2)
# NO diversity
# =========================
def grow_EA_2Obj_TwoTargets_QBC_ONLY(
    model1, model2,
    xTrainVal, yTrainVal,
    xTest, yTest,
    target1: str,
    target2: str,
    strSaveDir='',
    strSaveName='growing_EA_2obj_qbc_only',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
):
    """
    2-objective NSGA-II:
      maximize mean( minmax(|pred1 - pred2|) for target1 ),
      maximize mean( minmax(|pred1 - pred2|) for target2 ).
    NO diversity.

    Requires your existing:
      - splitTrainVal(...)
      - get_scores(...)
      - n_round, n_train_ratio
    """
    import random, time

    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]
    if target1 not in target_cols or target2 not in target_cols:
        raise ValueError(f"target1/target2 must be in: {target_cols}")

    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    lstTrainIndices = np.random.choice(xTrainVal.index, init_size, replace=False).tolist()

    # fixed test targets (for logging)
    yEC_test   = yTest[target_cols[0]]
    yS_test    = yTest[target_cols[1]]
    yZT_test   = yTest[target_cols[2]]
    yKtot_test = yTest[target_cols[3]]

    while True:
        t0 = time.time()

        yTV = yTrainVal[target_cols]
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTV, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.1%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        # unpack train targets
        yEC_train   = yTrain_temp[target_cols[0]]
        yS_train    = yTrain_temp[target_cols[1]]
        yZT_train   = yTrain_temp[target_cols[2]]
        yKtot_train = yTrain_temp[target_cols[3]]

        # model 1
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(model1, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val1,  s_mae_1,  s_rmse_1,  s_r2_1  = get_scores(model1, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(model1, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val1,  k_mae_1,  k_rmse_1,  k_r2_1  = get_scores(model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # model 2
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(model2, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val2,  s_mae_2,  s_rmse_2,  s_r2_2  = get_scores(model2, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(model2, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val2,  k_mae_2,  k_rmse_2,  k_r2_2  = get_scores(model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # log (same schema you already use)
        df1 = pd.concat([df1, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_1, n_round)], "rmse_elec_cond":[round(ec_rmse_1, n_round)], "r2_elec_cond":[round(ec_r2_1, n_round)],
            "mae_seebeck":[round(s_mae_1, n_round)],    "rmse_seebeck":[round(s_rmse_1, n_round)],    "r2_seebeck":[round(s_r2_1, n_round)],
            "mae_zt":[round(zt_mae_1, n_round)],        "rmse_zt":[round(zt_rmse_1, n_round)],        "r2_zt":[round(zt_r2_1, n_round)],
            "mae_k_total":[round(k_mae_1, n_round)],    "rmse_k_total":[round(k_rmse_1, n_round)],    "r2_k_total":[round(k_r2_1, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        df2 = pd.concat([df2, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_2, n_round)], "rmse_elec_cond":[round(ec_rmse_2, n_round)], "r2_elec_cond":[round(ec_r2_2, n_round)],
            "mae_seebeck":[round(s_mae_2, n_round)],    "rmse_seebeck":[round(s_rmse_2, n_round)],    "r2_seebeck":[round(s_r2_2, n_round)],
            "mae_zt":[round(zt_mae_2, n_round)],        "rmse_zt":[round(zt_rmse_2, n_round)],        "r2_zt":[round(zt_r2_2, n_round)],
            "mae_k_total":[round(k_mae_2, n_round)],    "rmse_k_total":[round(k_rmse_2, n_round)],    "r2_k_total":[round(k_r2_2, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        if len(xVal_temp) == 0:
            print("No validation candidates left. Stopping.")
            break

        # -------- compute QBC diffs (ABS), then min-max normalize per target --------
        pred_map_1 = {
            target_cols[0]: ec_pred_val1,
            target_cols[1]: s_pred_val1,
            target_cols[2]: zt_pred_val1,
            target_cols[3]: k_pred_val1,
        }
        pred_map_2 = {
            target_cols[0]: ec_pred_val2,
            target_cols[1]: s_pred_val2,
            target_cols[2]: zt_pred_val2,
            target_cols[3]: k_pred_val2,
        }

        diff1_abs = np.abs(pred_map_1[target1] - pred_map_2[target1])
        diff2_abs = np.abs(pred_map_1[target2] - pred_map_2[target2])

        q1 = minMaxStandarize(diff1_abs)
        q2 = minMaxStandarize(diff2_abs)

        diffs_df = pd.DataFrame({"q1": q1, "q2": q2}, index=yVal_temp.index)

        # -------- EA selection (2-objective ONLY, no diversity) --------
        to_add = EA_select_batch_twoQuality_only(
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(step_size, len(xVal_temp)),
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
        )

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices = lstTrainIndices + to_add
        print("Time to run loop:", round(time.time() - t0, 2), "seconds", flush=True)

    # save
    df1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False
    )
    return




# =========================
# 3-objective EA batch selection:
#   (quality1, quality2, quality3)
# NO diversity
# =========================
def EA_select_batch_threeQuality_only(
    candidates: pd.DataFrame,
    diffs_df: pd.DataFrame,          # columns ['q1','q2','q3'] aligned to candidates.index
    step_size: int,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
):
    """
    Objectives (maximize):
      1) mean q1 over batch
      2) mean q2 over batch
      3) mean q3 over batch

    Returns: list of candidates.index (length <= step_size)

    Requires your existing NSGA-II helper funcs:
      - _fast_non_dominated_sort, _crowding_distance, _tournament_select,
        _ordered_crossover, _make_unique, _mutate_swap
    """
    rng = np.random.default_rng(random_state)

    n_pool = len(candidates)
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    diffs_df = diffs_df.reindex(candidates.index)
    # accept either named columns or first 3 columns
    if not all(c in diffs_df.columns for c in ["q1", "q2", "q3"]):
        if diffs_df.shape[1] < 3:
            raise ValueError("diffs_df must have columns ['q1','q2','q3'] (or at least 3 columns).")
        diffs_df = diffs_df.iloc[:, :3]
        diffs_df.columns = ["q1", "q2", "q3"]

    q1 = np.nan_to_num(diffs_df["q1"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q2 = np.nan_to_num(diffs_df["q2"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q3 = np.nan_to_num(diffs_df["q3"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)
        return (
            float(np.mean(q1[idx])),
            float(np.mean(q2[idx])),
            float(np.mean(q3[idx])),
        )  # 3 objectives

    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,3)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # pick one solution from Pareto front
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,3)

    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)  # equal weights for (q1,q2,q3)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()


# =========================
# Grow loop using 3-objective EA:
#   objectives = normalized abs QBC diffs for (target1, target2, target3)
# NO diversity
# =========================
def grow_EA_3Obj_ThreeTargets_QBC_ONLY(
    model1, model2,
    xTrainVal, yTrainVal,
    xTest, yTest,
    target1: str,
    target2: str,
    target3: str,
    strSaveDir='',
    strSaveName='growing_EA_3obj_qbc_only',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
):
    """
    3-objective NSGA-II (NO diversity):
      maximize mean( minmax(|pred1 - pred2|) for target1 ),
      maximize mean( minmax(|pred1 - pred2|) for target2 ),
      maximize mean( minmax(|pred1 - pred2|) for target3 ).

    Requires your existing:
      - splitTrainVal(...)
      - get_scores(...)
      - n_round, n_train_ratio
      - EA helper funcs: _fast_non_dominated_sort, _crowding_distance, _tournament_select,
                        _ordered_crossover, _make_unique, _mutate_swap
    """
    import random, time

    target_cols = [
        "Electrical Conductivity (S/cm)",
        "Seebeck Coefficient (µV/K)",
        "zT",
        "Total Thermal Conductivity (W/mK)"
    ]
    for t in (target1, target2, target3):
        if t not in target_cols:
            raise ValueError(f"Targets must be in: {target_cols}")

    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    lstTrainIndices = np.random.choice(xTrainVal.index, init_size, replace=False).tolist()

    # fixed test targets (for logging)
    yEC_test   = yTest[target_cols[0]]
    yS_test    = yTest[target_cols[1]]
    yZT_test   = yTest[target_cols[2]]
    yKtot_test = yTest[target_cols[3]]

    while True:
        t0 = time.time()

        yTV = yTrainVal[target_cols]
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTV, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.1%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        # unpack train targets
        yEC_train   = yTrain_temp[target_cols[0]]
        yS_train    = yTrain_temp[target_cols[1]]
        yZT_train   = yTrain_temp[target_cols[2]]
        yKtot_train = yTrain_temp[target_cols[3]]

        # model 1
        ec_pred_val1, ec_mae_1, ec_rmse_1, ec_r2_1 = get_scores(model1, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val1,  s_mae_1,  s_rmse_1,  s_r2_1  = get_scores(model1, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val1, zt_mae_1, zt_rmse_1, zt_r2_1 = get_scores(model1, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val1,  k_mae_1,  k_rmse_1,  k_r2_1  = get_scores(model1, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # model 2
        ec_pred_val2, ec_mae_2, ec_rmse_2, ec_r2_2 = get_scores(model2, xTrain_temp, yEC_train,   xTest, yEC_test,   X_val=xVal_temp)
        s_pred_val2,  s_mae_2,  s_rmse_2,  s_r2_2  = get_scores(model2, xTrain_temp, yS_train,    xTest, yS_test,    X_val=xVal_temp)
        zt_pred_val2, zt_mae_2, zt_rmse_2, zt_r2_2 = get_scores(model2, xTrain_temp, yZT_train,   xTest, yZT_test,   X_val=xVal_temp)
        k_pred_val2,  k_mae_2,  k_rmse_2,  k_r2_2  = get_scores(model2, xTrain_temp, yKtot_train, xTest, yKtot_test, X_val=xVal_temp)

        # log (same schema you already use)
        df1 = pd.concat([df1, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_1, n_round)], "rmse_elec_cond":[round(ec_rmse_1, n_round)], "r2_elec_cond":[round(ec_r2_1, n_round)],
            "mae_seebeck":[round(s_mae_1, n_round)],    "rmse_seebeck":[round(s_rmse_1, n_round)],    "r2_seebeck":[round(s_r2_1, n_round)],
            "mae_zt":[round(zt_mae_1, n_round)],        "rmse_zt":[round(zt_rmse_1, n_round)],        "r2_zt":[round(zt_r2_1, n_round)],
            "mae_k_total":[round(k_mae_1, n_round)],    "rmse_k_total":[round(k_rmse_1, n_round)],    "r2_k_total":[round(k_r2_1, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        df2 = pd.concat([df2, pd.DataFrame({
            "train_ratio":[round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
            "mae_elec_cond":[round(ec_mae_2, n_round)], "rmse_elec_cond":[round(ec_rmse_2, n_round)], "r2_elec_cond":[round(ec_r2_2, n_round)],
            "mae_seebeck":[round(s_mae_2, n_round)],    "rmse_seebeck":[round(s_rmse_2, n_round)],    "r2_seebeck":[round(s_r2_2, n_round)],
            "mae_zt":[round(zt_mae_2, n_round)],        "rmse_zt":[round(zt_rmse_2, n_round)],        "r2_zt":[round(zt_r2_2, n_round)],
            "mae_k_total":[round(k_mae_2, n_round)],    "rmse_k_total":[round(k_rmse_2, n_round)],    "r2_k_total":[round(k_r2_2, n_round)],
            "train_size":[len(xTrain_temp)]
        })], ignore_index=True)

        if len(xVal_temp) == 0:
            print("No validation candidates left. Stopping.")
            break

        # -------- compute QBC diffs (ABS), then min-max normalize per target --------
        pred_map_1 = {
            target_cols[0]: ec_pred_val1,
            target_cols[1]: s_pred_val1,
            target_cols[2]: zt_pred_val1,
            target_cols[3]: k_pred_val1,
        }
        pred_map_2 = {
            target_cols[0]: ec_pred_val2,
            target_cols[1]: s_pred_val2,
            target_cols[2]: zt_pred_val2,
            target_cols[3]: k_pred_val2,
        }

        diff1_abs = np.abs(pred_map_1[target1] - pred_map_2[target1])
        diff2_abs = np.abs(pred_map_1[target2] - pred_map_2[target2])
        diff3_abs = np.abs(pred_map_1[target3] - pred_map_2[target3])

        q1 = minMaxStandarize(diff1_abs)
        q2 = minMaxStandarize(diff2_abs)
        q3 = minMaxStandarize(diff3_abs)

        diffs_df = pd.DataFrame({"q1": q1, "q2": q2, "q3": q3}, index=yVal_temp.index)

        # -------- EA selection (3-objective ONLY, no diversity) --------
        to_add = EA_select_batch_threeQuality_only(
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(step_size, len(xVal_temp)),
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
        )

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices = lstTrainIndices + to_add
        print("Time to run loop:", round(time.time() - t0, 2), "seconds", flush=True)

        # optional stop if fully used
        if len(lstTrainIndices) >= len(xTrainVal):
            print("Reached full training set. Stopping.")
            break

    # save
    df1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False
    )
    return
