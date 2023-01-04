import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt
import datetime as dt
import matplotlib.pyplot as plt


def get_prepare_data(Data_path: str, filename: str, cut_data: bool) -> pd.DataFrame:
    # This has to be changed for the hourly data !!!!!!!!!
    df = pd.read_csv(f"{Data_path}/{filename}", sep=";")
    df["Timestamp"] = df.apply(lambda row: dt.datetime(int(row.YYYY), int(row.MM), int(row.DD)), axis=1)
    df["Timestamp"] = pd.to_datetime(df.Timestamp)
    df.drop(["YYYY", "MM", "DD", "DOY"], axis=1, inplace=True)
    df = df.set_index("Timestamp")
    if cut_data:  
        df = df.iloc[0:100,]

    return df

def shift_values(df: pd.DataFrame):
    # change this to hours
    # when using this for the combined value, pay attetion that you don't shift the values about the gauge. Aber wenn wir das machen ist doch eh egal oder, weil es 
    # bleibt ja gleich, man muss halt drauf achten, dass es nur für das entsprechende gauge macht, nicht, dass sich das auf ein anderen gauge auswirkt
    prec = df["prec"].to_list() #shape (100,1)
    df_temp = df.drop(columns=["prec"], axis=1, inplace=False)
    shifted_1_day = df_temp.shift(1, axis=0).reset_index()
    shifted_2_day = df_temp.shift(2, axis=0).reset_index()
    shifted_3_day = df_temp.shift(3, axis=0).reset_index()

    temp = pd.merge(shifted_1_day, shifted_2_day, left_index=True, right_index=True, 
                    suffixes=("_1day", "_2day"))
    temp = temp.merge(shifted_3_day.add_suffix("_3day"), left_index=True, right_index=True)

    temp["prec"] = prec #(2320, xxx)
    # There are only NaNs values in the first few rows, resulting from shifting. Therefor it is no problem removing this data as we have so much data
    return temp.dropna()

def cv_score_average(df: pd.DataFrame, model_name: str, tss: TimeSeriesSplit, features: list[str], target: str):
    for train_idx, val_idx in tss.split(df):
        scores = []
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        match model_name:
            case "xgb":
                model = xgb.XGBRegressor(base_score=0.5, booster="gbtree", 
                                        n_estimators = 1000,
                                        objective="reg:squarederror",
                                        max_depth=3,
                                        learning_rate=0.01,
                                        verbosity = 0)
            case "RandomForrest":
                model = RandomForestRegressor(n_estimators=100)
            case "DecisionTree":
                model = DecisionTreeRegressor()
            case "SGD":
                model = SGDRegressor()
            case "LinearRegression":
                model = LinearRegression()
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test.to_list(), y_pred.tolist(), squared=False)
        scores.append(score)
        
        return np.average(scores)