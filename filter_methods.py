import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import logging
import sys
from logging_code import setup_logging
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

logger = setup_logging("filter_methods")


def fm(X_train_num, X_test_num, y_train, y_test):
    try:
        logger.info(f'Before Train columns : {X_train_num.shape} \n : {X_train_num.columns}')
        logger.info(f'Before Test columns : {X_test_num.shape} \n : {X_test_num.columns}')

        # Variance Threshold (remove low variance)
        reg = VarianceThreshold(threshold=0.01)
        X_train_var = reg.fit_transform(X_train_num)
        X_test_var = reg.transform(X_test_num)

        cols = X_train_num.columns[reg.get_support()]
        cols1 = X_train_num.columns[~reg.get_support()]

        X_train_var = pd.DataFrame(X_train_var, columns=cols, index=X_train_num.index)
        X_test_var = pd.DataFrame(X_test_var, columns=cols, index=X_test_num.index)

        logger.info(f'After Variance Threshold columns Number of good columns: {cols}')
        logger.info(f'After Variance Threshold columns Number of bad columns : {cols1}')

        # SelectKBest (Top Features)
        k = int(len(cols) * 0.7)  # keep 70% best features

        selector = SelectKBest(score_func=f_classif, k=k)

        X_train_sel = selector.fit_transform(X_train_var, y_train)
        X_test_sel = selector.transform(X_test_var)

        selected_cols = cols[selector.get_support()]

        X_train_final = pd.DataFrame(X_train_sel, columns=selected_cols, index=X_train_num.index)
        X_test_final = pd.DataFrame(X_test_sel, columns=selected_cols, index=X_test_num.index)

        logger.info(f'Selected Columns ({len(selected_cols)}) : {selected_cols}')

        logger.info(f'After Train columns : {X_train_final.shape}')
        logger.info(f'After Test columns : {X_test_final.shape}')

        return X_train_final, X_test_final

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')