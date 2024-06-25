from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.commons.utils import JsCode
from streamlit_echarts import st_pyecharts
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import streamlit as st
import wash_data
import get_data


# Author: Arnab Adhikary
def data_selected():
    '''
    This section is to get the dataset selected and transfer the dataset to plots function that works in different pages.
    '''
    x = st.sidebar.slider('The size of data:', 0.0, 1.0, 1.0, 0.01)
    df = wash_data.wash_data()
    is_graduate = st.sidebar.selectbox('Graduate', [None, True, False])
    is_married = st.sidebar.selectbox('Married', [None, True, False])
    is_female = st.sidebar.selectbox('Female', [None, True, False])
    is_self_employed = st.sidebar.selectbox('Self_employed', [None, True, False])
    is_urban = st.sidebar.selectbox('Urban', [None, True, False])
    credit_history = st.sidebar.selectbox('Credit_History', [None, True, False])
    df_selected = get_data.select_data(x, is_graduate, is_married, is_female, is_self_employed, is_urban,
                                       credit_history)
    return df_selected


# To implement the home page and make our dataset visible.
def page_home():
    df_selected = data_selected()
    # Display the welcome message in the center, bold, and larger font 
    st.markdown("<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>Welcome to our app</h1>",
                unsafe_allow_html=True)  # Display the DataFrame below the welcome message
    st.markdown(
        "<h2 style='font-weight: bold;'>APP Introduction:</h2>\n" "After analyzing the customer's own information, the customer will judge the likelihood of the success of the loan based on our analysis results.",
        unsafe_allow_html=True)
    st.markdown(
        "<h2 style='font-weight: bold;'>Dataset Source:</h2>\n" "About the company Dream Housing Finance Corporation. They have a presence in all urban, semi-urban and rural areas. They would like to present the relevant charts based on the details of the customer provided when filling out the online application form. These details include the borrower's gender, marital status, educational background, employment situation, income situation, co-applicant income, loan amount required, repayment time, number of loans, place of residence, etc. Here, they provide a partial data set.",
        unsafe_allow_html=True)
    st.dataframe(df_selected)
    st.markdown("""## The GUIDANCE

This APP is committed to showing the relationship between various aspects and whether the user can borrow according to the user's selection conditions. This app is a user page with strong interaction with users. Here's how to use it:
* The page has a sidebar for users to choose according to their own ideas, and this sidebar has three major blocks. The "Navigate" block allows the user to select different graphs (heat maps, bar charts, etc.) to show the relationship between each condition and the success of the loan, for example, clicking on the Plot_bar will show the average personal income of the male and female successful loan (in the data we collect). In the "The size of data" block, the user drags and drops the displayed line to select the cardinality of the dataset, for example, if you select 0.5, the displayed data analysis will be performed in the first half of the dataset we collected. The last block is to let the user select the condition to better query the information, for example, select "true" in "graduate", and the chart will only show the data analysis graph of the graduated population.
* The main page of the page is selected by the "Navigate" block. The "home" page is a brief introduction to the app and the publication of our dataset. The "Plot" page is a display of charts, and there are also blocks on the charts for users to select conditions, so as to show the data analysis graphs that users need in more detail. On the rest of the pages, there will be detailed questions that are of common interest to users, and users can follow the page introduction and get some data sheets.
""")  # Display the introduction text
    st.markdown('''## Conclusion

### Revenue:
1. The wage demand for men with successful loan conditions is slightly higher than that of women
2. Graduate loan success conditions have higher salary requirements than non-graduates
3. The salary requirement of unmarried users is slightly higher than that of married users
4. The wage demand of rural hukou users is slightly higher than that of urban hukou users
5. The wage demand of the successful conditions of the loan for the entrepreneurial group is higher than that of the working group users
6. The above conditions are similar to the relationship between the co-applicant's income\

### Loan Amount:
1. Men usually have more money to borrow than women
2. Graduates usually borrow more than non-graduates
3. Married users can usually borrow more than unmarried people
4. Rural accounts usually have more borrowing amounts than urban accounts
5. Entrepreneurs usually have more borrowing money than working people
6. People who do not have a loan record usually have more borrowing amount than users who have a record
7. The above conditions are similar to the relationship between the borrowing time and the duration of the loan
''')
    return None


def main():
    st.snow()
    # This section is to implement the control flow of our app, where the pages designing are implemented.
    session_state = st.session_state
    if 'page' not in session_state:
        session_state['page'] = 'Home'
    page = st.sidebar.radio('Navigate', ['Home'])
    # to implement multi-pages
    if page == 'Home':
        page_home()


main()
