import pandas as pd
import numpy as np
from sklearn import model_selection

def create_folds(data):
    data["kfold"]=-1

    data = data.sample(frac=1).reset_index(drop=True)
    
    
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f,(t,v) in enumerate(kf.split(X=df,y=data.Status.values)):
        df.loc[v,"kfold"] =  f

    df.to_csv("../input/data_folds.csv",index=False)



if __name__=="__main__":

    df = pd.read_excel("../input/Chapter 11 HR Analytics.xlsx",sheet_name="Data without missing values")

    df.drop(["SLNO","Candidate.Ref"],axis=1,inplace=True)

    df["Status"]=df["Status"].map({"Joined":1, "Not Joined":0})

    cat_cols = [ x for x in df.columns if df[x].dtype=="O"]

    for col in cat_cols:
        dummies = pd.get_dummies(df[col],drop_first=True,)
        df = pd.concat([df,dummies],axis=1)
        df.drop(col,axis=1,inplace=True)

    create_folds(df)

