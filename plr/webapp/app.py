from .. import predict_p
from flask import Flask, request, render_template, jsonify
import random
import pprint
import pickle

# load model and scaler
with open('../../lr_model.sav', 'rb') as f:
    model = pickle.load(f)

with open('../../scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# convert ajax response into suitable format


def get_value_for_feature(feature_name, response_data):
    '''Return value in response data that corresponds to a given feature name '''
    for item in response_data:
        if item['name'] == feature_name:
            val = item['value']
            if val == '':
                return 0
            else:
                return float(val)


def get_feature_values_array(feature_names, response_data):
    '''Return a list of values from response data in a correct order
    to use with predict_p function'''
    values = []
    for name in feature_names:
        values.append(get_value_for_feature(name, response_data))
    return values


app = Flask(__name__, static_url_path="")

feature_names = ['Monthly income', 'Gig economy weekly income',
                 'Atm check deposits weekly income', 'Direct deposit weekly income',
                 'Government aid weekly income', 'Frequency heuristic weekly income',
                 'Large deposits weekly income', 'Income length', 'Payrolls per month',
                 'Income sources', 'Days since most recent payroll',
                 'Days until next payroll', 'Bank balance', 'Overdraft transactions',
                 'NSF transactions', 'Return transactions', 'Reverse transactions',
                 'Mortgage', 'Auto loan', 'Student loan', 'Traditional single payment',
                 'Installment loan', 'Predatory installment loan', 'Title loan',
                 'Pay advances', 'Total loan payments', 'Nunmber of other active loans',
                 'Traditional single payment mean', 'Traditional single payment trend',
                 'Loan requested', 'State LA', 'State NH', 'State NJ']


@app.route('/')
def index():
    """Return home page."""
    return render_template('index.html')


@app.route('/make-predictions')
def make_predictions():
    """Return predictions page."""
    return render_template('make-predictions.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict_from_input():
    """Return prediction for a given user input."""
    data = request.get_json(force=True)

    prediction = predict_p(get_feature_values_array(feature_names, data),
                           model, scaler)[0][0]
    print(prediction, type(prediction))

    return jsonify({'probability': round(prediction, 2)})


@app.route('/contact')
def contact():
    """Return the contact page."""
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
