
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from src.visualization.visualize import Vis


training_data_path = r"D:\assignment2\assignment2-abdelmageed95\src\data\training_dataset.csv" 
def prepare_df(path):
    df = pd.read_csv(path)
    #drop the categorical columns 
    df.drop(['timestamp' , 'longest_word' ,'sld', ] ,axis = 1 , inplace = True )
    return df 



def training(df):
    y = df.iloc[ : ,-1]
    X = df.iloc[:, :-1] 
    # split data into training and validation
    X_train,X_test ,y_train, y_test = train_test_split(X,y,
                                                   test_size=0.1,
                                                   random_state=0,
                                                  shuffle = True,
                                                  stratify = y)                                                        
    rnd_clf = RandomForestClassifier(n_estimators=100,
                                    max_depth=12,
                                    criterion="gini",
                                    class_weight="balanced",
                                    bootstrap="true",
                                    random_state=1)
    rnd_clf.fit(X_train ,y_train)
    y_pred = rnd_clf.predict(X_test)
    Vis(rnd_clf ,X_test ,y_test ,y_pred)
    joblib.dump(rnd_clf,r"D:\assignment2\assignment2-abdelmageed95\models\random_forest.joblib")
    return "Done"
