#importing required libraries
from flask import Flask,request,render_template
import numpy as nu
import pandas as pa
from sklearn import metrics
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import featureextraction

file=open("pickle/model.pkl","rb")
gbc=pickle.load(file)
file.close()

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
     
      url=request.form["url"]
      obj=featureextraction(url)
      x = nu.array(obj.featureslist()).reshape(1,30) 


      y_pd=gbc.predict(x)[0]
     #1 is safe
     #-1 is unsafe
      y_pro_phishing = gbc.predict_proba(x)[0,0]
      y_pro_nonphishing = gbc.predict_proba(x)[0,1]
     #(if y_pd==1)
      pd = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
      return render_template('index.html',xx =round(y_pro_nonphishing,2),url=url )
    return render_template("index.html", xx =-1)

if __name__ == "__main__":
    app.run(debug=True)