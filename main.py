import streamlit as st
from st_pages.eda_page import main as eda_page
from st_pages.home_page import main as home_page
from st_pages.model_page import main as model_page
from st_pages.overview_page import main as overview_page

st.set_page_config(page_title='COVID-19 Case Prediction App', page_icon='assets/img/favicon.png', layout='wide')
st.write("""
<style>
div[data-testid="stMetric"]
{
    background-color: rgba(0, 0, 0, 0.5);
    color: white;
    padding: 10px 0 0 10px;
    border-radius: 5px;
    border-color: #26282e !important;
}
</style>
         
""", unsafe_allow_html=True)

home, overview, eda, model = st.tabs(['Home', 'Overview', 'EDA', 'Model'])

home_page(home)
overview_page(overview)
eda_page(eda)
model_page(model)