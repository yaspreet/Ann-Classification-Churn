import streamlit as st 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model = tf.keras.models.load_model('churn_model.h5')


with open('LabelEncoder.pkl','rb') as f :
    le = pickle.load(f)

with open('OneHotEncoder.pkl','rb') as f:
    ohe = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Prediction App')

geography = st.selectbox('Geography',ohe.categories_[0])
gender = st.selectbox('Gender',le.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
    'Geography':[geography]
})
input_data['Gender'] = le.transform(input_data['Gender'])
geo_encoded  = ohe.transform(input_data[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.drop('Geography',axis=1),geo_df],axis=1)

input_data_scaled = scaler.transform(input_data) 

prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

st.write(f'Churn Probability :{prediction_probab:.2f}')

if prediction_probab >0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction_probab}')
else:
    st.write(f'The customer is not likely to churn with a probability of {prediction_probab}')
    




