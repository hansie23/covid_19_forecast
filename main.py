import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from preprocessing import preprocess_differencing, preprocess_log

st.set_page_config(page_title="COVID-19 Forecast & Explorer", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("assets/data/preprocessed_data_updated.csv")
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

df = load_data()
case_model = load_model("assets/model/xgb_model_total_imputed_cases.pkl")
death_model = load_model("assets/model/xgb_model_total_deaths.pkl")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Explorer", "Case Prediction", "Death Prediction"])

if page == "Data Explorer":
    st.title("📊 COVID-19 Data Explorer")
    
    st.subheader("Dataset Overview")
    st.dataframe(df.head(100))
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download full dataset as CSV",
        data=csv,
        file_name='covid_data.csv',
        mime='text/csv',
    )
    
    st.subheader("Trends Over Time")
    metric = st.selectbox("Select Metric", ["imputed_total_cases", "imputed_total_deaths", "new_cases_smoothed", "new_deaths_smoothed"])
    fig = px.line(df, x="Date", y=metric, title=f"{metric.replace('_', ' ').title()} Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Correlations")
    corr_cols = st.multiselect("Select columns for correlation", df.select_dtypes(include=[np.number]).columns.tolist(), default=["imputed_total_cases", "imputed_total_deaths", "new_cases_smoothed", "new_deaths_smoothed"])
    if corr_cols:
        fig_corr = px.imshow(df[corr_cols].corr(), text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Case Prediction":
    st.title("🔮 Total Cases Forecast")
    
    with st.form("case_form"):
        col1, col2 = st.columns(2)
        with col1:
            fullyVaccinated = st.number_input("Fully Vaccinated", value=9327654)
            partiallyVaccinated = st.number_input("Partially Vaccinated", value=4663827)
            totalTests = st.number_input("Total Tests", value=239937354)
            totalVaccinations = st.number_input("Total Vaccinations", value=9982068)
            new_deaths_smoothed = st.number_input("New Deaths Smoothed", value=0.0)
        with col2:
            new_people_vaccinated_smoothed = st.number_input("New People Vaccinated Smoothed", value=0.0)
            new_vaccinations_smoothed = st.number_input("New Vaccinations Smoothed", value=0.0)
            stringency_index = st.number_input("Stringency Index", value=0.0)
            test24hours = st.number_input("Test 24 Hours", value=0)
            rfh = st.number_input("RFH (Rainfall)", value=0.0)
            r3h = st.number_input("R3H (Rainfall)", value=0.0)
            month = st.slider("Month", 1, 12, 1)
            day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 0)
        
        submit = st.form_submit_button("Predict Cases")
    
    if submit:
        input_df = pd.DataFrame({
            'fullyVaccinated': [fullyVaccinated],
            'new_deaths_smoothed': [new_deaths_smoothed],
            'new_people_vaccinated_smoothed': [new_people_vaccinated_smoothed],
            'new_vaccinations_smoothed': [new_vaccinations_smoothed],
            'partiallyVaccinated': [partiallyVaccinated],
            'stringency_index' : [stringency_index],
            'test24hours': [test24hours],
            'totalTests': [totalTests],
            'totalVaccinations': [totalVaccinations],
            'vaccinated24hours': [0.0], # Added as default since it was in original but not in main inputs
            'rfh': [rfh],
            'r3h': [r3h],
            'month':[month],
            'day_of_week': [day_of_week],
        })
        
        differenced_features = {
            'fullyVaccinated': 9327654, 'partiallyVaccinated': 4663827, 'totalTests': 239937354,
            'totalVaccinations': 9982068
        }
        differencing_data = pd.DataFrame([differenced_features])
        preprocessed_data = preprocess_differencing(input_df, differencing_data)
        
        prediction = case_model.predict(preprocessed_data)
        st.success(f"Predicted Total Imputed Cases: {prediction[0]:,.2f}")

elif page == "Death Prediction":
    st.title("⚰️ Total Deaths Forecast")
    
    with st.form("death_form"):
        col1, col2 = st.columns(2)
        with col1:
            imputed_active_cases = st.number_input("Imputed Active Cases", value=0.0)
            fullyVaccinated = st.number_input("Fully Vaccinated", value=9327654)
            partiallyVaccinated = st.number_input("Partially Vaccinated", value=4663827)
            totalVaccinations = st.number_input("Total Vaccinations", value=9982068)
            new_vaccinations_smoothed = st.number_input("New Vaccinations Smoothed", value=0.0)
        with col2:
            stringency_index = st.number_input("Stringency Index", value=13.89)
            total_tests_per_thousand = st.number_input("Total Tests per Thousand", value=180.0)
            positive_rate = st.number_input("Positive Rate", value=0.0)
            test24hours = st.number_input("Test 24 Hours", value=0)
            rfh = st.number_input("RFH (Rainfall)", value=0.0)
            r3h = st.number_input("R3H (Rainfall)", value=0.0)
            month = st.slider("Month", 1, 12, 1)
            day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 0)
        
        submit = st.form_submit_button("Predict Deaths")
        
    if submit:
        input_df = pd.DataFrame({
            'imputed_active_cases': [imputed_active_cases],
            'fullyVaccinated': [fullyVaccinated],
            'new_vaccinations_smoothed': [new_vaccinations_smoothed],
            'partiallyVaccinated': [partiallyVaccinated],
            'stringency_index' : [stringency_index],
            'test24hours': [test24hours],
            'totalVaccinations': [totalVaccinations],
            'total_tests_per_thousand': [total_tests_per_thousand],
            'vaccinated24hours': [0],
            'positive_rate': [positive_rate],
            'rfh': [rfh],
            'r3h': [r3h],
            'day_of_week': [day_of_week],
            'month':[month],
        })
        
        differenced_features = {
            'fullyVaccinated': 9327654, 'partiallyVaccinated': 4663827, 'stringency_index': 13.89,
            'totalVaccinations': 9982068
        }
        log_feature = {'total_tests_per_thousand': 180}
        
        differencing_data = pd.DataFrame([differenced_features])
        preprocessed_data = preprocess_differencing(input_df, differencing_data)
        
        for feature, val in log_feature.items():
            preprocessed_data[feature] = preprocess_log(preprocessed_data[feature], pd.Series([val]))
            
        prediction = death_model.predict(preprocessed_data)
        st.success(f"Predicted Total Deaths: {prediction[0]:,.2f}")
