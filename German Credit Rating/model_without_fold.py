import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
import joblib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("../input/cleaned_data.csv")

    X = df.drop("Credit Rating",axis=1).values
    y = df["Credit Rating"].values

    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.20, random_state=42,stratify = y
    )

    clf = tree.DecisionTreeClassifier()

    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)

    accuracy = metrics.accuracy_score(preds,y_test)
    roc_score = metrics.roc_auc_score(preds,y_test)
    

    joblib.dump("../model/",f"dt.bin")

    print(f"Accuracy {accuracy} ROC {roc_score}")
    print(metrics.classification_report(preds,y_test))







    

