from operator import index
import streamlit as st
# import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAToAAAChCAMAAABgSoNaAAAAclBMVEX///9HR0c4ODg+Pj6dnZ3JyclDQ0OioqI7OztnZ2exsbHz8/NAQEDHx8dOTk7MzMzl5eUzMzORkZHs7OxycnK5ubmIiIhgYGBYWFioqKj5+fng4ODv7+9aWlrY2Ni+vr56enp3d3ctLS2MjIwkJCQlJSVbBrScAAAJC0lEQVR4nO2d25aiOhBAFZSOigqKN/DW7cz//+LRpAK5Al19JGtN134bGmOygUpSCc5oRBAEQRAEQRAEQRAEQRAEQRDEv8p0EpoqtAIskzwOy9c0tAIsH/E4LBGpI3WkrjekDg2pQ0Pq0JA6NKQODalDQ+rQkDo0pA4NqUND6tCQOjSkDg2pQ0Pq0JA6NKQODalDQ+rQkDo0pA4NqUND6tCQOjSkDo2iLrVJrJYmvc76beoeG4u7aSU52CdtivqsROX3qFuV9l8XudFQNneUsWPwx1upsox+jbqF488H496JXBsJ58xpYf271U2MXiRxlVGrW2qHf7m6qd78dO8qg9S51J1TrZ1s5yqD1LnUjQot2OVb1zmkzqlOC3bJ0VkGqXOqu6rtTzfOMoKoK7Pti8xZ60HoUrdYqc1cuk4Jo+4U8XF3OsN9XOeK+VCXOi3YsbPzlCDqjlAv9vNXVq4X58Chi051J1a3MvFc4RDqrnKeE7uDSH/Oj4ihyuhUd23UxRN3GSHU7etRk6fefZmPY18M76BT3Uhp5drz7cOrK5vJdeQca/bkenjV/V3qPptRsTvUhVC3U8LIAVMA8ODlvEtdXcuk8JQRQJ2alshR/aNgk75TXSafDXbylDG8ukrUWvhL74gSgPeqq4Md813d4dWJOU4yE+5yfEfxZnUPCHaJIxfKGVxdKQZ18fQirqnvcejmzepuQoE31A2vTqbCRhtR+9X3iwDerO4sKurMrXMGV3cXDb7LQSdzTxB78GZ10J3l3hnP0OrOos7RbTS68Kqln98uA5i0qCur6fz0cbpdnZm2furEpXHm1jlDq4Pvy5vqx5nVroJjPClbcbR4ySjvRfF5FJU4fr6OapOl5X0cRyyOWcSSw8llr486HlrcuXW1KYOpE50Er1AmvtqeIu7ZK68SGcezNH1lW7jp8pgm2ioouzcn7uJIyY8nLN/b04E+6ravP7tz65yB1cHMX3wQxifWSqiY45pKM1Ah1Bmrfc34sJpZVU/jm1mPPup4/dy5dc7A6mCwJGzBXCcyG/YTdctEX5GBipp3tk+ddn8+R6D6THGhnTusulK0GJJNkAewJrI/UKcsvWubGCKjM/GpW6pPwJoZndBVC8vDqrvBfQZzG8g+mf1/H3Vxs13mtXUmuvNzKjnxTFh6LIpDwuRJud7p+NTNVTkLU8spoDqRtq5vM/gWsxfrVjfaZtlWnJYcztkTHpLqm5Edd9xIOa0jX64126tO6xVmaaKFukL716DqMnGXNZMvaGqqh+oe6l7AuEt53CdyA82mee5kpjzVJlQ+dTvtKp4iLbdejsOp+xBfltath+VOY7KDVbeV95wWoG7Sp5rs9apj6r+uX1pJ1Z9g6koxf1CWSSrx7UZHgVUHV8ZchoH7TstvedX91ez80XLr83Dq1jCoUwYjsGanp/+x6uDxz83pyQXCgtJwr7ovbaT0qQ1W9l/B1EGHGikDABja6R0FUh10r/ak2DF+9KrLtZroCzrG+HhAdQuHpgWEp1itFFIdNMVeqy+hI1cCl1dd5N5f8qLKg6mDi59rl/JudrojtDo5SrSzujMzxrZ0E2lLkimYOghF+mWt98Uox5DqjvqgUUGKso/Y6vzz/SIJpQ5CUVLsNGL7+5Hq3OPrFzA+Ue7HlrvOm2WKgsU6SKg/50gaUJT6NCHViUvj2uYArVFa7leXXDzLONdg6spVR1nNkAKnDnbsxx/OVnN1zVf41Y2ZlXoVPE0FUtf1OoGydQen7gzqHEts31PnWcd5DkEDqbu78mgadRvc6s6rdnX/113nCXavpyaMum2nuaZrc6uT0zifutIf62ABU1kub1HnfH9H7BoLo04mMFy/DC3+0nQUHnViGQfTw8rpRJ8e1jGRqxsQRh3s0YkdV7SCG1ImQEGduUq4HXeogwfaMa6bWKvlbercwe7TWqgYSB0s57jjCCzv1H8UCxjmuXKbj1fd3Z4iA+K6qVsg2tQ5txDxfdn91H21/PR6nfLvrw6Wc3LnBsk6owatACdGC6CLRsxhF+KTagRoUzeOHXXkPU0vddWm5ff+693d/dU5plsNJbRDTmRhL5Tx5AlTLergzk6szInDaas6114JHqt7qWul3vDYW93Nvu4qcj/WWKmmuZFHDqlb8nVQiBXmYfKcKMm3VnWuYMejwc/Vzb+tTm6m82Ql5EIWZDxBtJYRlylgRR3cnE06AXoDM0ss0/p39VibOse+sFLUf3h1lbtNDdD/wrNWOXa23eQjb6pTgkAmH3ytb17KFvZbm9CLlIhgEECdXL3xbkOUJYqtOzCCUxScN3WP36g7iVJTnvzl3epDuns0vewcjvVcEeMVsffAihgyvDq5PJp6JtbN+6cQDGXsiy+7Z2XL62bcvCjYqJMhNy329wv/4FmeFMM67GhdfG8dFtRZ11j0jMOrW3dtLW3eQjny26Wqn07G0jRm6iSuUVe/9JOkSSwe9WWz+s9WRXFJm9V/fWNLuzq7pqLc4dWBl5b9VvUQHJZeHs2rFTWpORFbqEIP4hE9KUN5bc+JcR91xDpz+9PVSviNBlEnn8akZbM6zO3l9S4v1q/fRBsR2xp19WI/B0YeJ9c0KLGGGx3qIiPYQVgdXN3cMTiwkG2Bn8VYFPpPjST5ZGSpG80ad3WmdDq2rR+tsN+hzgx2cnA1tLpLyt9+zVtPziJxVvSQjVg1ezNT3viP1//9xSI13N9z2O+UNEvP5SZnzZP6/OzY0bF3qDNmJLC3zafuG1nL76nLZmIPcMd7w/vCPG26PyYsiliy2vMh2fqDoy3IXzeX14s/q9lDGW0v5rMxE3uJU/isSYc6Je/Kv5m1qnvrbAJLdn3iHdIIzlm2tZIl5+tyfjrN15VnjUb5TaeZykUGiIN69CiPzlxHx9q57dSf+Rd+NsHzW1bfOaofbqX+yD+hLgykjtSRuv6QOjSkDg2pQ0Pq0JA6NKQODalDQ+rQkDo0pA4NqUND6tCQOjSkDg2pQ0Pq0JA6NKQODalDQ+rQkDo0pA4NqUND6tCQOjSkDs3kKwrLX/QP44amWk7DsvT8dwIEQRAEQRAEQRAEQRAEQRDEL+U/jdO7yNi5iEQAAAAASUVORK5CYII=")
    st.title("AutoML")
    st.info("Build and explore your data.")
    choice = st.radio("NAVIGATION", ["Upload","Data Analysis","Modeling", "Download"])


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Data Analysis":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modeling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")


