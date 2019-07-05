import random
#import pandas as pd
from flask import Flask, request, render_template, jsonify
from .. import predict

app = Flask(__name__, static_url_path="")


@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_from_input():
    """Return a random prediction."""
    data = request.json
    prediction = predict_p([data['user_input']])
    return jsonify({'probability': prediction[0][1]})


if __name__ == '__main__':
    app.run(debug=True)
