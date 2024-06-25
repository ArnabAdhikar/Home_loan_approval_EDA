import pandas as pd
import numpy as np
# Author: Arnab Adhikary
# This section is to wash the data, making it more convenient for EDA. And I use some build-in functions to pad the nan
# value of data we chose.


def is_graduate(x):
    if x == 'Graduate':
        return 1
    else:
        return 0


def is_female(x):
    if x == 'Female':
        return 1
    else:
        return 0


def is_married(x):
    if x == 'Yes':
        return 1
    else:
        return 0


def is_urban(x):
    if x == 'Urban':
        return 1
    else:
        return 0


def is_self_employed(x):
    if x == 'Yes':
        return 1
    else:
        return 0


def Loan_Status_(x):
    if x == 'Y':
        return 1
    else:
        return 0


def wash_data():
    HomeLoansApproval = pd.read_csv('loan_sanction_train.csv')
    # drop the data that has same ID
    HomeLoansApproval = HomeLoansApproval.drop_duplicates(subset=['Loan_ID'])
    # use subset to point out the columns whose nan values are deleted
    HomeLoansApproval.dropna(axis=0, how='any',
                             subset=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History'],
                             inplace=True)
    HomeLoansApproval_mean = HomeLoansApproval['LoanAmount'].fillna(value=HomeLoansApproval['LoanAmount'].mean(),
                                                                    inplace=False)
    # median replacing
    HomeLoansApproval_median = HomeLoansApproval['Loan_Amount_Term'].fillna(
        HomeLoansApproval['Loan_Amount_Term'].median(), inplace=False)
    # replace the old series object with new series.
    HomeLoansApproval = HomeLoansApproval.drop(labels=['LoanAmount', 'Loan_Amount_Term'], axis=1)
    HomeLoansApproval['LoanAmount'] = HomeLoansApproval_mean
    HomeLoansApproval['Loan_Amount_Term'] = HomeLoansApproval_median
    HomeLoansApproval['Is_graduate'] = HomeLoansApproval['Education'].apply(lambda x: is_graduate(x))
    HomeLoansApproval['Is_Female'] = HomeLoansApproval['Gender'].apply(lambda x: is_female(x))
    HomeLoansApproval['Is_married'] = HomeLoansApproval['Married'].apply(lambda x: is_married(x))
    HomeLoansApproval['Is_urban'] = HomeLoansApproval['Property_Area'].apply(lambda x: is_urban(x))
    HomeLoansApproval['Is_self_employed'] = HomeLoansApproval['Self_Employed'].apply(lambda x: is_self_employed(x))
    HomeLoansApproval['Loan_Status_'] = HomeLoansApproval['Loan_Status'].apply(lambda x: Loan_Status_(x))
    Loan_Status = HomeLoansApproval['Loan_Status_']
    HomeLoansApproval = HomeLoansApproval.drop(
        ['Education', 'Gender', 'Married', 'Property_Area', 'Self_Employed', 'Loan_Status', 'Loan_Status_'], axis=1)
    HomeLoansApproval['Loan_Status'] = Loan_Status
    HomeLoansApproval['Dependents'] = HomeLoansApproval['Dependents'].apply(
        lambda x: ((0 if x == '0' else 1) if x != '2' else 2) if x != '3+' else 3)
    return HomeLoansApproval
