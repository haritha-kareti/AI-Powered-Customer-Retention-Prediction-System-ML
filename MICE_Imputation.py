import numpy as np
import pandas as pd
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from logging_code import setup_logging
logger = setup_logging("MICE_Imputation")


def handle_missing_value(X_train, X_test):
    try:
        logger.info(f"Before Handling Null values X_train shape: {X_train.shape} \n {X_train.isnull().sum()}")
        logger.info(f"Before Handling Null values X_test shape: {X_test.shape} \n {X_test.isnull().sum()}")

        # Separate numerical columns
        num_cols = X_train.select_dtypes(include=np.number).columns

        # Initialize MICE Imputer
        imputer = IterativeImputer(random_state=42)

        # Fit on train and transform both
        X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = imputer.transform(X_test[num_cols])

        logger.info(f"After Handling Null values X_train shape: {X_train.shape} \n {X_train.isnull().sum()}")
        logger.info(f"After Handling Null values X_test shape: {X_test.shape} \n {X_test.isnull().sum()}")

        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")