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

feature_names = ['monthly_income', 'gig_economy_weekly_income',
                 'atm_check_deposits_weekly_income', 'direct_deposit_weekly_income',
                 'government_aid_weekly_income', 'frequency_heuristic_weekly_income',
                 'large_deposits_weekly_income', 'income_length', 'payrolls_per_month',
                 'income_sources', 'days_since_most_recent_payroll',
                 'days_until_next_payroll', 'bank_balance', 'overdraft_transactions',
                 'nsf_transactions', 'return_transactions', 'reverse_transactions',
                 'mortgage', 'auto_loan', 'student_loan', 'traditional_single_payment',
                 'installment_loan', 'predatory_installment_loan', 'title_loan',
                 'pay_advances', 'total_loan_payments', 'no_of_other_active_loans',
                 'traditional_single_payment_mean', 'traditional_single_payment_trend',
                 'loan_requested', 'state_"ID"', 'state_"UT"', 'state_"WA"']


@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict_from_input():
    """Return prediction for a given user input."""
    data = request.get_json(force=True)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)
    pp.pprint(get_feature_values_array(feature_names, data))
    prediction = predict_p(get_feature_values_array(feature_names, data),
                           model, scaler)[0][0]
    print(prediction, type(prediction))

    return jsonify({'probability': prediction})


if __name__ == '__main__':
    app.run(debug=True)
