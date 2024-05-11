from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')
from feature import featureextraction

app = Flask(__name__)

try:
    file = open("pickle/model.pkl", "rb")
    gbc = pickle.load(file)
    file.close()
except ModuleNotFoundError as e:
    # Handle the ModuleNotFoundError gracefully
    print(f"Error loading model: {e}")
    gbc = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if gbc is None:
            return "Model not available. Please check the server configuration."
        
        url = request.form["url"]
        obj = featureextraction(url)
        x = np.array(obj.featureslist()).reshape(1, 30) 

        y_pd = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_nonphishing = gbc.predict_proba(x)[0, 1]
        
        pd_result = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)
        
        return render_template('index.html', xx=round(y_pro_nonphishing, 2), url=url, pd_result=pd_result)
    
    return render_template("index.html", xx=-1)


