from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():
	mass = float(request.form['mass'])
	width = float(request.form['width'])
	height = float(request.form['height'])
	color_score = float(request.form['color_score'])

	arr = np.array([[mass, width, height, color_score]])
	pred = model.predict(arr)
	return render_template("prediction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)
