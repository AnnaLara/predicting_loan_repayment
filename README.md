# predicting_loan_repayment

### Business understanding

I have been working with a startup in Seattle that grants small loans (up to 500\$) to people with no credit score.

Loans for people with no credit score is an unexplored business opportunity, as there are around 50000 people in Uniteed States without credit score. On one hand, this startup helps this people granting them a loan when noone else would not, and on the other hand, the startup reports information.

I have built a Logistic Regression model that predicts the probability of default and paying back.

### Data Understanding

The data consisted of 2 datasets: 4145 approved applications with labels, 2927 rejected applications.

There were 36 bank transaction features:
`monthly_income`, `gig_economy_weekly_income`, `atm_check_deposits_weekly_income`, `direct_deposit_weekly_income`, `government_aid_weekly_income`, `frequency_heuristic_weekly_income`,`large_deposits_weekly_income`, `income_length`, `payrolls_per_month`, `income_sources`, `days_since_most_recent_payroll`, `days_until_next_payroll`, `bank_balance`, `overdraft_transactions`, `nsf_transactions`, `return_transactions`, `reverse_transactions`,
`mortgage`, `auto_loan`, `student_loan`, `traditional_single_payment`,`installment_loan`, `predatory_installment_loan`, `title_loan`, `pay_advances`, `total_loan_payments` `no_of_other_active_loans`, `traditional_single_payment_mean`, `traditional_single_payment_trend`,
`loan_requested`, `state`, `loan_id`, `loan_amount`, `loan_borrowed_inc`, `loan_outstanding_inc`,
`length_of_transaction_history`.

### Data Preparation

The data required minimal cleaning and preparation. I had to complete the following steps:

1. Dropped `loan_id`.

2. I dropeed those columns that were directly connected with the loan granted or contained values that directly correlated with loan status (paid back or no), since the idea was to create model that evaluates new applications before granting the loan. Dropped columns: `loan_amount`, `loan_borrowed_inc`, `loan_outstanding_inc`.

3. I dropped `length_of_transaction_history`because of its high correlation with `ìncome_length` feature.

4. Encoding `state` columns into categories using sklearn's `OneHotEncoder`. I chose 4 states with the most observations that could be found in both approved applications and rejected applications datasets. Since there were not many rows with applications from other states, I dropped those rows. I also droppped rows where state was not indicated.

5. Scaling: Unscaled values produced inconsistent results in the logistic regression model, so I scaled all but categorical columns to use in Logistic regression.

### Modeling

**My baseline** was a model that guesses the majority class 1 (paying back the loan). Cross-validated AUC score of such model is 0.5

I applied different models such as Logistic Regression, Random Forests, Gradient Boosting, XGBoost. I built two Logistic Regression models: with all the features and only with features with low p-values. Redcung features did not improve the results.

After the first iteration, since my label classes were unbalanced (75% class 1, 25% class 0), I applied two different teqniques to balance classes:

- randomly picking data from rejected applications dataset which I assumed is class 0 for the purposes of the experiment.

- upsampling class 0 applications using `sklearn.resample``

In both cases I applied all the above mentioned models to compare the results.

**Comparisom of the applied models**
|**index**|**0.0**| **1.0**| **accuracy**| **CV_AUC_score**| **model**|
| --- | --- | --- | --- | --- | --- |
|0| {'precision': 0.35, 'recall': 0.03, 'f1': 0.06}| {'precision': 0.78, 'recall': 0.98, 'f1': 0.87} |0.77| 0.625946| Logistic regression|
|1| {'precision': 0.1, 'recall': 0.01, 'f1': 0.01}| {'precision': 0.78, 'recall': 0.99, 'f1': 0.87} |0.77| 0.646706| Logistic Regression limited features|
|2| {'precision': 0.67, 'recall': 0.06, 'f1': 0.12} |{'precision': 0.79, 'recall': 0.99, 'f1': 0.88}| 0.78| 0.643459| Random forest|
|3| {'precision': 0.44, 'recall': 0.04, 'f1': 0.08}| {'precision': 0.78, 'recall': 0.98, 'f1': 0.87}| 0.77| 0.653591 |XGBoost|
|4| {'precision': 0.36, 'recall': 0.03, 'f1': 0.05} |{'precision': 0.78, 'recall': 0.99, 'f1': 0.87}| 0.77| 0.654784| Gradient boosting|
|5| {'precision': 0.32, 'recall': 0.42, 'f1': 0.36}| {'precision': 0.82, 'recall': 0.74, 'f1': 0.78}| 0.67| 0.643459| Random forest with data from rejected dataset|
|6| {'precision': 0.3, 'recall': 0.44, 'f1': 0.36}| {'precision': 0.81, 'recall': 0.7, 'f1': 0.75}| 0.64| 0.627792| Logistic regression with data from rejected dataset|
|7| {'precision': 0.32, 'recall': 0.39, 'f1': 0.35} |{'precision': 0.81, 'recall': 0.76, 'f1': 0.78}| 0.68| 0.654784| Gradient Boosting with data from rejected dataset|
|8| {'precision': 0.33, 'recall': 0.4, 'f1': 0.36}| {'precision': 0.82, 'recall': 0.77, 'f1': 0.79}| 0.69| 0.653591| XG Boost with data from rejected dataset|
|9| {'precision': 0.38, 'recall': 0.13, 'f1': 0.19} |{'precision': 0.79, 'recall': 0.94, 'f1': 0.86}| 0.76| 0.643459| Random forest with resampled data|
|10| {'precision': 0.32, 'recall': 0.49, 'f1': 0.39} |{'precision': 0.83, 'recall': 0.71, 'f1': 0.76}| 0.66| 0.654784| Gradient Boosting with resampled data|
|11| {'precision': 0.33, 'recall': 0.49, 'f1': 0.4}| {'precision': 0.83, 'recall': 0.72, 'f1': 0.77}| 0.67| 0.653591| XG Boost with resampled data|
|12| {'precision': 0.33, 'recall': 0.66, 'f1': 0.44}| {'precision': 0.86, 'recall': 0.61, 'f1': 0.71}| 0.62| 0.627920| Logistic regression with resampled data|

Baseline AUC was 0.5

The models had quite similar AUC score, which in this case is the most important metric. Desicion tree based models have slightly better AUC, but they tend to overfit and are not calibrated as is logistic regression. For this reason the selected model had to be a logistic regression based one.

There are several logistic regression models. The ones with blanced classes seem to have better balanced predictive power for classes 0 and 1. There is no much difference betweer logistic regression upsampled from rejected daraset and the one upsampled using sklearn resample. I selected the last logistic regression upsampled with sklearn resample as a final model.

### Evaluation

To evaluate the results I would make simple logistic regression with just income/expenses ratio. I will be evaluating the results using CV score as well as F1 score, precision and recall.

### Deployment

I will deploy the model as a Flask app
