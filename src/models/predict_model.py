import joblib
from src.features import build_features
import pandas as pd 

""" 
function to load model and 
"""
load_model = joblib.load(r"D:\assignment2\assignment2-abdelmageed95\models\random_forest.joblib")


def predict_fun(row_df, loaded_rf):
    r = row_df.drop(['longest_word' ,'sld', ] ,axis = 1 )
    x_test = r.values
    label = loaded_rf.predict(x_test)
    confidence_score = round(loaded_rf.predict_proba(x_test)[0][label[0]], 2)
    return label , confidence_score

def generate_df(url):
    df = build_features.construct_df(url)
    label ,confidence_score = predict_fun(df,load_model)
    df['label'] = label
    df['confidence_score'] = confidence_score
    df.insert(0 , "url" , url)
    return df   

def save_data(output , output_path):
    Out_df = pd.concat(output , ignore_index= True)
    Out_df.to_csv(output_path + "\Out_df.csv")
    return 