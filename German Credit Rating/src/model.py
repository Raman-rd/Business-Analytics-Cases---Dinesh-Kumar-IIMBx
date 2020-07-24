import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
import joblib

def run(fold):

    df = pd.read_csv("../input/data_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("Credit Rating" , axis=1).values
    y_train = df_train["Credit Rating"].values

    x_valid = df_valid.drop("Credit Rating",axis=1).values
    y_valid = df_valid["Credit Rating"].values


    clf = tree.DecisionTreeClassifier()

    clf.fit(x_train,y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(preds,y_valid)
    roc_score = metrics.roc_auc_score(preds,y_valid)

    joblib.dump("../model/",f"dt{fold}.bin")

    print(f"fold {fold} Accuracy {accuracy} ROC {roc_score}")


if __name__=="__main__":
    run(fold=0)
    run(fold=1)
    



