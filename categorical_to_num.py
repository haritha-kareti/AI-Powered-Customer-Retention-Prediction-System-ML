import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import logging
import sys
from logging_code import setup_logging
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = setup_logging("categorical_to_num")


def c_t_n(X_train_cat, X_test_cat):
    try:
        logger.info(f'Before X_train_cat : {X_train_cat.shape} \n {X_train_cat.columns}')
        logger.info(f'Before X_test_cat : {X_test_cat.shape} \n {X_test_cat.columns}')


        #  Ordinal Columns (with order)

        ordinal_cols = ['Contract', 'InternetService']

        categories = [
            ['Month-to-month', 'One year', 'Two year'],  # Contract
            ['No', 'DSL', 'Fiber optic']                # Internet
        ]

        od = OrdinalEncoder(categories=categories)

        X_train_ord = pd.DataFrame(
            od.fit_transform(X_train_cat[ordinal_cols]),
            columns=[col + "_ord" for col in ordinal_cols],
            index=X_train_cat.index
        )

        X_test_ord = pd.DataFrame(
            od.transform(X_test_cat[ordinal_cols]),
            columns=[col + "_ord" for col in ordinal_cols],
            index=X_test_cat.index
        )

        # Drop ordinal cols
        X_train_cat = X_train_cat.drop(ordinal_cols, axis=1)
        X_test_cat = X_test_cat.drop(ordinal_cols, axis=1)

        # Remaining → OneHot

        oh = OneHotEncoder(drop='first', handle_unknown='ignore')

        oh.fit(X_train_cat)

        X_train_oh = pd.DataFrame(
            oh.transform(X_train_cat).toarray(),
            columns=oh.get_feature_names_out(X_train_cat.columns),
            index=X_train_cat.index
        )

        X_test_oh = pd.DataFrame(
            oh.transform(X_test_cat).toarray(),
            columns=oh.get_feature_names_out(X_test_cat.columns),
            index=X_test_cat.index
        )

        # Combine

        X_train_final = pd.concat([X_train_ord, X_train_oh], axis=1)
        X_test_final = pd.concat([X_test_ord, X_test_oh], axis=1)

        logger.info(f'After Encoding Train : {X_train_final.shape} \n {X_train_final.columns}')
        logger.info(f'After Encoding Test : {X_test_final.shape} \n {X_test_final.columns}')

        logger.info(f'Train Null values : {X_train_final.isnull().sum()}')
        logger.info(f'Test Null values : {X_test_final.isnull().sum()}')

        return X_train_final, X_test_final

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')