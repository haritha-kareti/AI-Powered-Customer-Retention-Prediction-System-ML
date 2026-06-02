import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
import sys
from logging_code import setup_logging
logger = setup_logging("variable_outliers")
from scipy.stats import yeojohnson


def vt_outliers(X_train_num, X_test_num):
    try:
        logger.info(f"Before Train Column Name : {X_train_num.columns}")
        logger.info(f"Before Test Column Name : {X_test_num.columns}")

        X_train_new = pd.DataFrame(index=X_train_num.index)
        X_test_new = pd.DataFrame(index=X_test_num.index)

        for i in X_train_num.columns:

            #  Yeo-Johnson (fit on train only)
            X_train_yeo, lam_value = yeojohnson(X_train_num[i])

            # Apply SAME lambda to test
            X_test_yeo = yeojohnson(X_test_num[i], lmbda=lam_value)

            # Convert to Series
            X_train_yeo = pd.Series(X_train_yeo, index=X_train_num.index)
            X_test_yeo = pd.Series(X_test_yeo, index=X_test_num.index)

            # IQR Capping (Winsorization)
            Q1 = X_train_yeo.quantile(0.25)
            Q3 = X_train_yeo.quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Apply capping
            X_train_cap = np.clip(X_train_yeo, lower_limit, upper_limit)
            X_test_cap = np.clip(X_test_yeo, lower_limit, upper_limit)

            # Store results
            X_train_new[i + "_trans"] = X_train_cap
            X_test_new[i + "_trans"] = X_test_cap

        logger.info(f"After Train Column Name : {X_train_new.columns}")
        logger.info(f"After Test Column Name : {X_test_new.columns}")

        return X_train_new, X_test_new

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')