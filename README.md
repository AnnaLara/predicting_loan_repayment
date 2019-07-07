# predicting_loan_repayment



### Business understanding



I have been working with a startup in Seattle that grants small loans (up to 500$) to people with no credit score.

Loans for people with no credit score is an unexplored business opportunity, as there are around 50000 people in Uniteed States without credit score. On one hand, this startup helps this people granting them a loan when noone else would not, and on the other hand, the startup reports information 

I am going to attempt to predict loan repayment using the data they have on customers' bank transactions.

Predicting if the person will pay the loan is a cornerstone task for the sratup I work with. This startup not only gives poeple a chance granting a loan when noone else would, but they also report this information to the credit authorities contributing to creating a credit score.

### Data Understanding

The data that will be handled for me will contain information in customers' bank transaction together with the ground thruth labels.

### Data Preparation

I expect to need to group transactions into categories, separating income and expenses transactions. I would also group transactions by day of the week and month. One more option would be to create groups of expenses by amount.

It seems that the data will include information on how a customer chose to repay the loan (installments/other) and how he/she actually paid it. This could be be used as a part of an advanced approach depending on the time it takes to do the initial model.

### Modeling

The first model is going to be logistic regresstion. I also plan to apply random forests and gradient boosting technique.

### Evaluation

To evaluate the results I would make simple logistic regression with just income/expenses ratio. I will be evaluating the results using CV score as well as F1 score, precision and recall.

### Deployment

I will deploy the model as a Flask app