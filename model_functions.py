import xgboost as xgb
import json
import pickle
import numpy as np



def LoadTrainedModel(FilePath:str):
    model = xgb.XGBRegressor()
    model.load_model(FilePath)
    return model


def LoadEncoder(FilePath:str):
    LoadedEncoder = pickle.load(open(FilePath, 'rb'))
    return LoadedEncoder


def LoadLambda(FilePath:str):
    with open(FilePath, "r") as f:
        data = json.load(f)
    return data["lambda"]


def LoadUniqueValues(FilePath:str):
    with open(FilePath, "r") as f:
        data = json.load(f)
    return data

def Encoder(X, OHenc, OrdEncoder):
    OHencX = np.array([[X["Customer"], X["Category"]]])
    OrdEncX = np.array([[X["Product"], X["Region"]]])
    RemainingX = np.array([[X["Quantity"]]])
    OHencX = OHenc.transform(OHencX)
    OrdEncX = OrdEncoder.transform(OrdEncX)
    NewArr = np.hstack((RemainingX, OHencX, OrdEncX))
    return NewArr
