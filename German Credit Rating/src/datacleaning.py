import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


if __name__=="__main__":

    df = pd.read_excel("../input/Chapter 12 German Credit Rating.xlsx")

    df = df.drop("S.No",axis=1)

    cat_cols = [ f for f in df.columns if df[f].dtypes=="O"]

    for f in cat_cols:
        enc = LabelEncoder()
        enc.fit(df[f])
        df[f] = enc.transform(df[f])

        df["Credit Amount"] = np.log(df["Credit Amount"]+1)

        df["Age"] = np.log(df["Age"]+1)


        df.to_csv("../input/cleaned_data.csv")