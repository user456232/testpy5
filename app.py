import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask,request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

def fit():
    data = pd.read_csv("Titanic_Cleaned_data.csv")
    data=pd.DataFrame(data)
    X=data.iloc[:,2:9]
    y=data.iloc[:,1:2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model_rf = RandomForestClassifier(n_estimators=10)
    model_rf.fit(X_train, y_train)


#inputt=[int(x) for x in "1 0 35 1 0 53 0".split(' ')]
#final=[np.array(inputt)]

#b = model_rf.predict(final)

def predict(input):
    final=[np.array(inputt)]
    return model_rf.predict(final)


@app.route('/',methods=['POST'])
def index():
    #p=np.array(list(request.get_json().values()))
    #inputes=[request.get_json()["pclass"],request.get_json()["sex"],request.get_json()["age"],request.get_json()["sibsp"],
    #        request.get_json()["parch"],request.get_json()["fare"],request.get_json()["alone"]]
    #p=predict(inputes)[0]
   # print(p)
    return "request.get_json()"


if __name__ == "__main__":
    CORS(app)
    app.run(debug=False)