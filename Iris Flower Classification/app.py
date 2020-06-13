import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app= Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("iris.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if(prediction==0):
        output="SETOSA"
    elif(prediction==1):
        output='VERSICOLOR'
    else:
        output='VIRGINICA'
    return render_template('iris.html',prediction_text="Predicted Flower is {}".format(str(output)))

if __name__ == "__main__":
    app.run(debug=True)
