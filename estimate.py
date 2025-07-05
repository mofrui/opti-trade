import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import(
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000
ins_is = 0

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="./prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))
# prcALL[ins_id]
# print(prcAll[0, :].reshape(-1, 1))

def evaluation():
    pass

def pred():
    pass

