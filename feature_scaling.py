import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import logging
import sys
import pickle
from logging_code import setup_logging
from sklearn.preprocessing import RobustScaler
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

logger = setup_logging("feature_scaling")


def fs(X_train, y_train, X_test, y_test):
    try:
        logger.info(f'Training data independent size : {X_train.shape}')
        logger.info(f'Training data dependent size : {y_train.shape}')
        logger.info(f'Testing data independent size : {X_test.shape}')
        logger.info(f'Testing data dependent size : {y_test.shape}')
        logger.info(f'Before Scaling : {X_train.head(1)}')

        # Scaling
        sc = RobustScaler()
        sc.fit(X_train)

        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        # Save scaler
        with open('robust_scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)

        common(X_train_sc, y_train, X_test_sc, y_test,)

        # LogisticRegression Model
        reg = LogisticRegression()
        reg.fit(X_train_sc, y_train)
        y_pred = reg.predict(X_test_sc)

        #  Evaluation
        logger.info(f'Accuracy : {accuracy_score(y_test, y_pred)}')
        logger.info(f'Confusion Matrix : \n{confusion_matrix(y_test, y_pred)}')
        logger.info(f'Classification Report : \n{classification_report(y_test, y_pred)}')


        # Save Model
        with open('Model.pkl', 'wb') as t:
            pickle.dump(reg, t)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')