import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns


def run(fold):

    df = pd.read_csv("../input/data_folds.csv")

    df_train = df[df.kfold!=fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("Status",axis=1).values
    y_train = df_train.Status.values

    x_valid = df_valid.drop("Status",axis=1).values
    y_valid = df_valid.Status.values

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.fit_transform(x_valid)
    clf = ensemble.RandomForestClassifier()
    clf.fit(x_train,y_train)

    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid,preds)

    roc_score = metrics.roc_auc_score(y_valid,preds)

    joblib.dump("../model/",f"dt{fold}.bin")

    metrics.plot_roc_curve(clf,x_train,y_train)

    print(f"Fold={fold} ROC={roc_score} Accuracy {accuracy}")
    print(metrics.classification_report(y_valid,preds))

if __name__=="__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
