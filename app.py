import streamlit as st 
from streamlit_option_menu import  option_menu
from streamlit_toggle import st_toggle_switch
import streamlit.components.v1 as com
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit_authenticator as stauth
from deta import Deta
from datetime import date
import datetime
from database import *
from streamlit_extras.stoggle import stoggle
from markdownlit import mdlit
import pickle
from pathlib import Path
from streamlit_echarts import st_echarts



def redirect(_url):
    link=''
    st.markdown(link, unsafe_allow_html=True)
st.set_page_config(layout="wide",page_title='VINTERN', page_icon="👩‍🎓")
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 2rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 10rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
#MainMenu {visibility: hidden;}
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
selected=option_menu(
    menu_title=None,
    options=["Sign up","Info","Projects","Dashboard"],
    orientation="horizontal",key="BSDFDS"
)

if selected=="Sign up":
    placeholder = st.empty()
    email = st.text_input("Email",autocomplete = None)
    st.markdown(" ")
    name=st.text_input("User name",autocomplete = None) 
    st.markdown(" ")
    password = st.text_input("Password",autocomplete = None)  
    if st.button("Sign Up",key="option_tab2"): 
        db.put({"key":email, "name":name, "password": password})
          #insertuser(email,name,password) 
        st.success("Succesful Registered")
import json
import requests
def month3():
   st.markdown("3Months Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3ao3ek7w25iwT" async> </script> </form>
   """,height=600)         

def month2():
   st.markdown("2Months Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3atmhUZoprfiD" async> </script> </form>
   """,height=600)         

def month12():
   st.markdown("45 Day's Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3ax3c6WXm3krs" async> </script> </form>
   """,height=600)         

def month1():
   st.markdown("1 Month Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3bipYZTFUmUPj" async> </script> </form>
   """,height=600)            

def month6():
   st.markdown("6 Month Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3borb1TpmAcGT" async> </script> </form>
   """,height=600)    
   
   
def monthh3():
   st.markdown("3Months Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L5HuZSUl7KxixT" async> </script> </form>
   """,height=600)         

def monthh2():
   st.markdown("2Months Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L5Hx4ujHGO9Lsw" async> </script> </form>
   """,height=600)         

def monthh12():
   st.markdown("45 Day's Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3ax3c6WXm3krs" async> </script> </form>
   """,height=600)         

def monthh1():
   st.markdown("1 Month Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3bipYZTFUmUPj" async> </script> </form>
   """,height=600)            

def monthh6():
   st.markdown("6 Month Internship")   
   com.html("""
     <form><script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_L3borb1TpmAcGT" async> </script> </form>
   """,height=600)              
def rpey():    
      col1, col2, col3,col4,col5 = st.tabs(["1 MONTH","45 DAY'S","2 MONTH'S","3 MONTH'S","6 MONTH'S"])
      with col1:
         month1()
      with col2:
         month12()
      with col3:
         month2()
      with col4:
         month3()
      with col5:
         month6()  
         
def rpuy():    
      col1, col2, col3,col4,col5 = st.tabs(["1 MONTH","45 DAY'S","2 MONTH'S","3 MONTH'S","6 MONTH'S"])
      with col1:
         monthh1()
      with col2:
         monthh12()
      with col3:
         monthh2()
      with col4:
         monthh3()
      with col5:
         monthh6()           
         
def pay():
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="11")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
      
            
#### 
def food():
   stoggle(
      "RobotChef:",
      """[VI12001] RobotChef - Refining recipes based on user reviews.""",
   )
   stoggle(
      "Food Amenities:",
      """[VI12002] Food Amenities - Predicting the demand for food amenities using neural networks""",
   )
   stoggle(
      "Recipe Cuisine and Rating:",
      """[VI12003] Recipe Cuisine and Rating - Predict the rating and type of cuisine from a list of ingredients""",
   )
   stoggle(
      "Food Classification:",
      """[VI12004] Food Classification - Classification using Keras.""",
   )
   stoggle(
      "Image to Recipe:",
      """[VI12005] Image to Recipe - Translate an image to a recipe using deep learning.""",
   )
   stoggle(
      "Calorie Estimation:",
      """[VI12006] Calorie Estimation - Estimate calories from photos of food""",
   )
   stoggle(
      "Fine Food Reviews:",
      """[VI12007] Fine Food Reviews - Sentiment analysis on Amazon Fine Food Reviews.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="12")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def restaurant():
   stoggle(
      "Restaurant Violation:",
      """[VI12008]Restaurant Violation - Food inspection violation forecasting""",
   )
   stoggle(
      "Restaurant Success:",
      """[VI12009]Restaurant Success - Predict whether a restaurant is going to fail""",
   )
   stoggle(
      "Predict Michelin",
      """[VI12010]Predict Michelin - Predict the likelihood that restaurant is a Michelin restaurant.""",
   )
   stoggle(
      "Restaurant Inspection",
      """[VI12011]Restaurant Inspection - An inspection analysis to see if cleanliness is related to rating.""",
   )                  
   stoggle(
      "Sales",
      """[VI12012]Sales - Restaurant sales forecasting with LTSM.""",
   )
   stoggle(
      "Visitor Forecasting",
      """[VI12013]Visitor Forecasting - Reservation and visitation number prediction.""",
   )
   stoggle(
      "Restaurant Profit",
      """[VI12014]Restaurant Profit - Restaurant regression analysis.""",
   )
   stoggle(
      "Competition",
      """[VI12015]Competition - Restaurant competitiveness analysis.""",
   )
   stoggle(
      "Business Analysis",
      """[VI12016]Business Analysis - Restaurant business analysis project.""",
   )
   stoggle(
      "Location Recommendation",
      """[VI12017]Location Recommendation - Restaurant location recommendation tool and analysis.""",
   )
   stoggle(
      "Closure, Rating and Recommendation",
      """[VI12018]Closure, Rating and Recommendation - Three prediction tasks using Yelp data.""",
   )
   stoggle(
      "Anti-recommender",
      """[VI12019]Anti-recommender - Find restaurants you don’t want to attend.""",
   )
   stoggle(
      "Menu Analysis",
      """[VI12020]Menu Analysis - Deeper analysis of restaurants through their menus.""",
   )
   stoggle(
      "Menu Recommendation",
      """[VI12021]Menu Recommendation - NLP to recommend restaurants with similar menus.""",
   )
   stoggle(
      "Food Price",
      """[VI12022]Food Price - Predict food cost.""",
   )
   stoggle(
      "Automated Restaurant Report",
      """[VI12023]Automated Restaurant Report - Automated machine learning company report.""",
   )
   stoggle(
      "Peer-to-Peer Housing",
      """[VI12024]Peer-to-Peer Housing - The effect of peer to peer rentals on housing.""",
   )
   stoggle(
      "Roommate Recommendation",
      """[VI12025]Roommate Recommendation - A system for students seeking roommates.""",
   )
   stoggle(
      "Room Allocation",
      """[VI12026]Room Allocation - Room allocation process.""",
   )
   stoggle(
      "Dynamic Pricing",
      """[VI12027]Dynamic Pricing - Hotel dynamic pricing calculations.""",
   )
   stoggle(
      "Hotel Similarity",
      """[VI12028]Hotel Similarity - Compare brands that directly compete""",
   )   
   stoggle(
      "Hotel Reviews",
      """[VI12029]Hotel Reviews - Cluster hotel reviews.""",
   )                  
   stoggle(
      "Predict Prices",
      """[VI12030]Predict Prices - Predict hotel room rates.""",
   )              
   stoggle(
      "Hotels vs Airbnb",
      """[VI12031]Hotels vs Airbnb - Comparing the two approaches.""",
   )              
   stoggle(
      "Hotel Improvement",
      """[VI12032]Hotel Improvement - Analyse reviews to suggest hotel improvements.""",
   )              
   stoggle(
      "Orders",
      """[VI12033]Orders - Order cancellation prediction for hotels.""",
   )              
   stoggle(
      "Fake Reviews",
      """[VI12034]Fake Reviews - Identify whether reviews are fake/spam.""",
   )              
   stoggle(
      "Reverse Image Lodging",
      """[VI12035]Reverse Image Lodging - Find your preferred lodging by uploading an image.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="13")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()   
   
def ml():
   stoggle(
      "Chart of Account Prediction",
      """[VI12036]Chart of Account Prediction - Using labeled data to suggest the account name for every transaction""",
   )          
   stoggle(
      "Accounting Anomalies",
      """[VI12037]Accounting Anomalies - Using deep-learning frameworks to identify accounting anomalies""",
   )                      
   stoggle(
      "Financial Statement Anomalies",
      """[VI12038]Financial Statement Anomalies - Detecting anomalies before filing, using R""",
   )            
   stoggle(
      "Useful Life Prediction (FirmAI)",
      """[VI12039]Useful Life Prediction (FirmAI) - Predict the useful life of assets using sensor observations and feature engineering""",
   )            
   stoggle(
      "AI Applied to XBRL",
      """[VI12040]AI Applied to XBRL - Standardized representation of XBRL into AI and Machine learning""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="14")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()   
          
def analytics():
   stoggle(
      "Forensic Accounting",
      """""",
   )         
   stoggle(
      "General Ledger (FirmAI)",
      """""",
   )         
   stoggle(
      "Bullet Graph (FirmAI)",
      """""",
   )         
   stoggle(
      "Aged Debtors (FirmAI)",
      """""",
   )         
   stoggle(
      "Automated FS XBRL",
      """""",
   )         
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="15")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()   
   
def Textual_analytics():
   stoggle(
      "Financial Sentiment Analysis",
      """[VI12046]Financial Sentiment Analysis - Sentiment, distance and proportion analysis for trading signals.""",
   )       
   stoggle(
      "Extensive NLP",
      """[VI12047]Extensive NLP - Comprehensive NLP techniques for accounting research.""",
   )    
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="16")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()        

def parse():
   stoggle(
      "EDGAR",
      """[VI12048]EDGAR - A walk-through in how to obtain EDGAR data.""",
   )  
   stoggle(
      "IRS",
      """[VI12049]IRS - Acessing and parsing IRS filings.""",
   )  
   stoggle(
      "Financial Corporate",
      """[VI12050]Financial Corporate - Rutgers corporate financial datasets.""",
   )  
   stoggle(
      "Non-financial Corporate",
      """[VI12051]Non-financial Corporate - Rutgers non-financial corporate dataset.""",
   )  
   stoggle(
      "PDF Parsing",
      """[VI12052]PDF Parsing - Extracting useful data from PDF documents.""",
   )  
   stoggle(
      "PDF Tabel to Excel",
      """[VI12053]PDF Tabel to Excel - How to output an excel file from a PDF.""",
   )      
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="17")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()    
    
def economics():
   stoggle(
      "Prices",
      """[VI12054]Prices - Agricultural price prediction.""",
   )
   stoggle(
      "Prices 2",
      """[VI12055]Prices 2 - Agricultural price prediction.""",
   )
   stoggle(
      "Yield",
      """[VI12056]Yield - Agricultural analysis looking at crop yields in Ukraine.""",
   )
   stoggle(
      "Recovery",
      """[VI12057]Recovery - Strategic land use for agriculture and ecosystem recovery""",
   )
   stoggle(
      "MPR",
      """[VI12058]MPR - Mandatory Price Reporting data from the USDA's Agricultural Marketing Service.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="18")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
     
def development():
   stoggle(
      "Segmentation",
      """[VI12059]Segmentation - Agricultural field parcel segmentation using satellite images.""",
   )
   stoggle(
      "Water Table",
      """[VI12060]Water Table - Predicting water table depth in agricultural areas.""",
   )
   stoggle(
      "Assistant",
      """[VI12061]Assistant - Notebooks from agricultural assistant.""",
   )
   stoggle(
      "Eco-evolutionary",
      """[VI12062]Eco-evolutionary - Eco-evolutionary dynamics.""",
   )
   stoggle(
      "Diseases",
      """[VI12063]Diseases - Identification of crop diseases and pests using Deep Learning framework from the images.""",
   )
   stoggle(
      "Irrigation and Pest Prediction",
      """[VI12064]Irrigation and Pest Prediction - Analyse irrigation and predict pest likelihood.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="19")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def consumer():
   stoggle(
      "Loan Acceptance",
      """[VI12065]Loan Acceptance - Classification and time-series analysis for loan acceptance.""",
   )
   stoggle(
      "Predict Loan Repayment",
      """[VI12066]Predict Loan Repayment - Predict whether a loan will be repaid using automated feature engineering.""",
   )
   stoggle(
      "Loan Eligibility Ranking",
      """[VI12067]Loan Eligibility Ranking - System to help the banks check if a customer is eligible for a given loan.""",
   )
   stoggle(
      "Home Credit Default (FirmAI)",
      """[VI12068]Home Credit Default (FirmAI) - Predict home credit default.""",
   )
   stoggle(
      "Mortgage Analytics",
      """[VI12069]Mortgage Analytics - Extensive mortgage loan analytics.""",
   )
   stoggle(
      "Credit Approval",
      """[VI12070]Credit Approval - A system for credit card approval.""",
   )
   stoggle(
      "Loan Risk",
      """[VI12071]Loan Risk - Predictive model to help to reduce charge-offs and losses of loans.""",
   )
   stoggle(
      "Amortisation Schedule (FirmAI)",
      """[VI12072]Amortisation Schedule (FirmAI) - Simple amortisation schedule in python for personal use.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="20")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def management():
   stoggle(
      "Credit Card",
      """[VI12073]Credit Card - Estimate the CLV of credit card customers.""",
   )
   stoggle(
      "Survival Analysis",
      """[VI12074]Survival Analysis - Perform a survival analysis of customers.""",
   )
   stoggle(
      "Next Transaction",
      """[VI12075]Next Transaction - Deep learning model to predict the transaction amount and days to next transaction.""",
   )
   stoggle(
      "Credit Card Churn",
      """[VI12076]Credit Card Churn - Predicting credit card customer churn.""",
   )
   stoggle(
      "Bank of England Minutes",
      """[VI12077]Bank of England Minutes - Textual analysis over bank minutes.""",
   )
   stoggle(
      "CEO",
      """[VI12078]CEO - Analysis of CEO compensation.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="21")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def valuation():   
   stoggle(
      "Zillow Prediction",
      """[VI12079]Zillow Prediction - Zillow valuation prediction as performed on Kaggle.""",
   )
   stoggle(
      "Real Estate",
      """[VI12080]Real Estate - Predicting real estate prices from the urban environment.""",
   )
   stoggle(
      "Used Car",
      """[VI12081]Used Car - Used vehicle price prediction.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="22")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
      
def fraud():   
   stoggle(
      "XGBoost",
      """[VI12082]XGBoost - Fraud Detection by tuning XGBoost hyper-parameters with Simulated Annealing""",
   )
   stoggle(
      "Fraud Detection Loan in R",
      """[VI12083]Fraud Detection Loan in R - Fraud detection in bank loans.""",
   )
   
   stoggle(
      "AML Finance Due Diligence",
      """[VI12084]AML Finance Due Diligence - Search news articles to do finance AML DD.""",
   )
   stoggle(
      "Credit Card Fraud",
      """[VI12085]Credit Card Fraud - Detecting credit card fraud.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="23")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
      
def risk():   
   stoggle(
      "Car Damage Detective",
      """[VI12086]Car Damage Detective - Assessing car damage with convolution neural networks for a personal auto claims.""",
   )
   stoggle(
      "Medical Insurance Claims",
      """[VI12087]Medical Insurance Claims - Predicting medical insurance claims.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="24")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
      
def anomaly():   
   stoggle(
      "Claim Denial",
      """[VI12088]Claim Denial - Predicting insurance claim denial""",
   )
   
   stoggle(
      "Claim Fraud",
      """[VI12089]Claim Fraud - Predictive models to determine which automobile claims are fraudulent.""",
   )
   stoggle(
      "Claims Anomalies",
      """[VI12090]Claims Anomalies - Anomaly detection system for medical insurance claims data.""",
   )
   stoggle(
      "Actuarial Sciences (R)",
      """[VI12091]Actuarial Sciences (R) - A range of actuarial tools in R.""",
   )
   stoggle(
      "Bank Failure",
      """[VI12092]Bank Failure - Predicting bank failure.""",
   )
   stoggle(
      "Risk Management",
      """[VI12093]Risk Management - Finance risk engagement course resources.""",
   )
   
   stoggle(
      "VaR GaN",
      """[VI12094]VaR GaN - Estimate Value-at-Risk for market risk management using Keras and TensorFlow.""",
   )
   stoggle(
      "Compliance",
      """[VI12095]Compliance - Bank Grievance Compliance Management.""",
   )
   stoggle(
      "Stress Testing",
      """[VI12096]Stress Testing - ECB stress testing.""",
   )
   stoggle(
      "Stress Testing Techniques",
      """[VI12097]Stress Testing Techniques - A notebook with various stress testing exercises.""",
   )
   stoggle(
      "Reverse Stress Test",
      """[VI12098]Reverse Stress Test - Given a portfolio and a predefined loss size, determine which factors stress (scenarios) would lead to that loss""",
   )
   
   stoggle(
      "BoE stress test",
      """[VI12099]BoE stress test- Stress test results and plotting.""",
   )
   stoggle(
      "Recovery",
      """[VI12100]Recovery - Recovery of money owed.""",
   )
   stoggle(
      "Quality Control",
      """[VI12101]Quality Control - Quality control for banking using LDA""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="25")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
      
def physical():   
   stoggle(
      "Bank Note Fraud Detection",
      """[VI12102]Bank Note Fraud Detection - Bank Note Authentication Using DNN Tensorflow Classifier and RandomForest.""",
   )
   stoggle(
      "ATM Surveillance",
      """[VI12103]ATM Surveillance - ATM Surveillance in banks use case.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="26")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()   
   
def general():
   stoggle(
      "Programming",
      """[VI12104]Programming - Python Programming for Biologists""",
   )  
   stoggle(
      "Introduction DL",
      """[VI12105]Introduction DL - A Primer on Deep Learning in Genomics""",
   )   
   stoggle(
      "Pose",
      """[VI12106]Pose - Estimating animal poses using DL.""",
   )   
   stoggle(
      "Privacy",
      """[VI12107]Privacy - Privacy preserving NNs for clinical data sharing.""",
   )   
   stoggle(
      "Population Genetics",
      """[VI12108]Population Genetics - DL for population genetic inference.""",
   )   
   stoggle(
      "Bioinformatics Course",
      """[VI12109]Bioinformatics Course - Course materials for Computational Biologyand Bioinformatics""",
   )   
   stoggle(
      "Applied Stats",
      """[VI12110]Applied Stats - Applied Statistics for High-Throughput Biology""",
   )   
   stoggle(
      "Scripts",
      """[VI12111]Scripts - Python scripts for biologists.""",
   )   
   stoggle(
      "Molecular NN",
      """[VI12112]Molecular NN - A mini-framework to build and train neural networks for molecular biology.""",
   )   
   stoggle(
      "Systems Biology Simulations",
      """[VI12113]Systems Biology Simulations - Systems biology practical on writing simulators with F# and Z3""",
   )   
   stoggle(
      "Cell Movement",
      """[VI12114]Cell Movement - LSTM to predict biological cell movement.""",
   )   
   stoggle(
      "Deepchem",
      """[VI12115]Deepchem - Democratizing Deep-Learning for Drug Discovery, Quantum Chemistry, Materials Science and Biology""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="27")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def sequencing():     
   stoggle(
      "DNA, RNA and Protein Sequencing",
      """[VI12116]DNA, RNA and Protein Sequencing - Anew representation for biological sequences using DL.""",
   )   
   stoggle(
      "CNN Sequencing",
      """[VI12117]CNN Sequencing - A toolbox for learning motifs from DNA/RNA sequence data using convolutional neural networks""",
   )   
   stoggle(
      "NLP Sequencing",
      """[VI12118]NLP Sequencing - Language transfer learning model for genomics""",
   )   
   stoggle(
      "Chemoinformatics and drug discovery",
      """[VI12119]Chemoinformatics and drug discovery""",
   )   
   stoggle(
      "Novel Molecules",
      """[VI12120]Novel Molecules - A convolutional net that can learn features.""",
   )   
   stoggle(
      "Automating Chemical Design",
      """[VI12121]Automating Chemical Design - Generate new molecules for efficient exploration.""",
   )   
   stoggle(
      "GAN drug Discovery",
      """[VI12122]GAN drug Discovery - A method that combines generative models with reinforcement learning.""",
   )   
   stoggle(
      "RL",
      """[VI12123]RL - generating compounds predicted to be active against a biological target.""",
   )   
   stoggle(
      "One-shot learning",
      """[VI12124]One-shot learning - Python library that aims to make the use of machine-learning in drug discovery straightforward and convenient.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="28")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def genomics():     
   stoggle(
      "Jupyter Genomics",
      """[VI12125]Jupyter Genomics - Collection of computation biology and bioinformatics notebooks.""",
   )   
   stoggle(
      "Variant calling",
      """[VI12126]Variant calling - Correctly identify variations from the reference genome in an individual's DNA.""",
   )   
   stoggle(
      "Gene Expression Graphs",
      """[VI12127]Gene Expression Graphs - Using convolutions on an image.""",
   )   
   stoggle(
      "Autoencoding Expression",
      """[VI12128]Autoencoding Expression - Extracting relevant patterns from large sets of gene expression data""",
   )   
   stoggle(
      "Gene Expression Inference",
      """[VI12129]Gene Expression Inference - Predict the expression of specified target genes from a panel of about 1,000 pre-selected “landmark genes”.""",
   )   
   stoggle(
      "Plant Genomics",
      """[VI12130]Plant Genomics - Presentation and example material for Plant and Pathogen Genomics""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="29")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def life_sciences():    
   stoggle(
      "Plants Disease",
      """[VI12131]Plants Disease - App that detects diseases in plants using a deep learning model.""",
   )   
   stoggle(
      "Leaf Identification",
      """[VI12132]Leaf Identification - Identification of plants through plant leaves on the basis of their shape, color and texture.""",
   )   
   stoggle(
      "Crop Analysis",
      """[VI12133]Crop Analysis - An imaging library to detect and track future position of ears on maize plants""",
   )   
   stoggle(
      "Seedlings",
      """[VI12134]Seedlings - Plant Seedlings Classification from kaggle competition""",
   )   
   stoggle(
      "Plant Stress",
      """[VI12135][Plant Stress](http://An ontology containing plant stresses; biotic and abiotic.) - An ontology containing plant stresses; biotic and abiotic.""",
   )   
   stoggle(
      "Animal Hierarchy",
      """[VI12136]Animal Hierarchy - Package for calculating animal dominance hierarchies.""",
   )   
   stoggle(
      "Animal Identification",
      """[VI12137]Animal Identification - Deep learning for animal identification.""",
   )   
   stoggle(
      "Species",
      """[VI12138]Species - Big Data analysis of different species of animals""",
   )   
   stoggle(
      "Animal Vocalisations",
      """[VI12139]Animal Vocalisations - A generative network for animal vocalizations""",
   )   
   stoggle(
      "Evolutionary",
      """[VI12140]Evolutionary - Evolution Strategies Tool""",
   )   
   stoggle(
      "Glaciers",
      """[VI12141]Glaciers - Educational material about glaciers.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="30")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def student():      
   stoggle(
      "Student Performance",
      """[VI12142]Student Performance - Mining student performance using machine learning.""",
   )   
   stoggle(
      "Student Performance 2",
      """[VI12143]Student Performance 2 - Student exam performance.""",
   )   
   stoggle(
      "Student Performance 3",
      """[VI12144]Student Performance 3 - Student achievement in secondary education.""",
   )   
   stoggle(
      "Student Performance 4",
      """[VI12145]Student Performance 4 - Students Performance Evaluation using Feature Engineering""",
   )   
   stoggle(
      "Student Intervention",
      """[VI12146]Student Intervention - Building a student intervention system.""",
   )   
   stoggle(
      "Student Enrolment",
      """[VI12147]Student Enrolment - Student enrolment and performance analysis.""",
   )   
   stoggle(
      "Academic Performance",
      """[VI12148]Academic Performance - Explore the demographic and family features that have an impact a student's academic performance.""",
   )   
   stoggle(
      "Grade Analysis",
      """[VI12149]Grade Analysis - Student achievement analysis.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="31")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def school():     
   stoggle(
      "School Choice",
      """[VI12150]School Choice - Data analysis for education's school choice.""",
   )   
   stoggle(
      "School Budgets and Priorities",
      """[VI12151]School Budgets and Priorities - Helping the school board and mayor make strategic decisions regarding future school budgets and priorities""",
   )   
   stoggle(
      "School Performance",
      """[VI12152]School Performance - Data analysis practice using data from data.utah.gov on school performance.""",
   )   
   stoggle(
      "School Performance 2",
      """[VI12153]School Performance 2 - Using pandas to analyze school and student performance within a district""",
   )   
   stoggle(
      "School Performance 3",
      """[VI12154]School Performance 3 - Philadelphia School Performance""",
   )   
   stoggle(
      "School Performance 4",
      """[VI12155]School Performance 4 - NJ School Performance""",
   )   
   stoggle(
      "School Closure",
      """[VI12156]School Closure - Identify schools at risk for closure by performance and other characteristics.""",
   )   
   stoggle(
      "School Budgets",
      """[VI12157]School Budgets - Tools and techniques for school budgeting.""",
   )   
   stoggle(
      "School Budgets",
      """[VI12158]School Budgets - Same as a above, datacamp.""",
   )   
   stoggle(
      "PyCity",
      """[VI12159]PyCity - School analysis.""",
   )      
   stoggle(
      "PyCity 2 ",
      """[VI12160]PyCity 2 - School budget vs school results.""",
   )      
   stoggle(
      "Budget NLP",
      """[VI12161]Budget NLP - NLP classification for budget resources.""",
   )      
   stoggle(
      "Budget NLP 2",
      """[VI12162]Budget NLP 2 - Further classification exercise.""",
   )      
   stoggle(
      "Budget NLP 3",
      """[VI12163]Budget NLP 3 - Budget classification.""",
   )      
   stoggle(
      "Survey Analysis",
      """[VI12164]Survey Analysis - Education survey analysis.""",
   )   
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="32")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def preventive():      
   stoggle(
      "Emergency Mapping",
      """[VI12165]Emergency Mapping - Detection of destroyed houses in California""",
   )      
   stoggle(
      "Emergency Room",
      """[VI12166]Emergency Room - Supporting emergency room decision making""",
   )      
   stoggle(
      "Emergency Readmission",
      """[VI12167]Emergency Readmission - Adjusted Risk of Emergency Readmission.""",
   )      
   stoggle(
      "Forest Fire",
      """[VI12168]Forest Fire - Forest fire detection through UAV imagery using CNNs""",
   )      
   stoggle(
      "Emergency Response",
      """[VI12169]Emergency Response - Emergency response analysis.""",
   )      
   stoggle(
      "Emergency Transportation",
      """[VI12170]Emergency Transportation - Transportation prompt on emergency services""",
   )      
   stoggle(
      "Emergency Dispatch",
      """[VI12171]Emergency Dispatch - Reducing response times with predictive modeling, optimization, and automation""",
   )      
   stoggle(
      "Emergency Calls",
      """[VI12172]Emergency Calls - Emergency calls analysis project.""",
   )      
   stoggle(
      "Calls Data Analysis",
      """[VI12173]Calls Data Analysis - 911 data analysis.""",
   )      
   stoggle(
      "Emergency Response",
      """[VI12174]Emergency Response - Chemical factory RL.""",
   )      
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="33")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
      
def crime():   
   stoggle(
      "Crime Classification",
      """[VI12175]Crime Classification - Times analysis of serious assaults misclassified by LAPD.""",
   )      
   stoggle(
      "Article Tagging",
      """[VI12176]Article Tagging - Natural Language Processing of Chicago news article""",
   )      
   stoggle(
      "Crime Analysis",
      """[VI12177]Crime Analysis - Association Rule Mining from Spatial Data for Crime Analysis""",
   )      
   stoggle(
      "Chicago Crimes",
      """[VI12178]Chicago Crimes - Exploring public Chicago crimes data set in Python""",
   )      
   stoggle(
      "Graph Analytics",
      """[VI12179]Graph Analytics - The Hague Crimes.""",
   )      
   stoggle(
      "Crime Prediction",
      """[VI12180]Crime Prediction - Crime classification, analysis & prediction in Indore city.""",
   )      
   stoggle(
      "Crime Prediction2",
      """[VI12181]Crime Prediction - Developed predictive models for crime rate.""",
   )      
   stoggle(
      "Crime Review",
      """[VI12182]Crime Review - Crime review data analysis.""",
   )      
   stoggle(
      "Crime Trends",
      """[VI12183]Crime Trends - The Crime Trends Analysis Tool analyses crime trends and surfaces problematic crime conditions""",
   )      
   stoggle(
      "Crime Analytics",
      """[VI12184]Crime Analytics - Analysis of crime data in Seattle and San Francisco.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="34")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def ambulance():        
   stoggle(
      "Ambulance Analysis",
      """[VI12185]Ambulance Analysis - An investigation of Local Government Area ambulance time variation in Victoria.""",
   )      
   stoggle(
      "Site Location",
      """[VI12186]Site Location - Ambulance site locations.""",
   )      
   stoggle(
      "Dispatching",
      """[VI12187]Dispatching - Applying game theory and discrete event simulation to find optimal solution for ambulance dispatching""",
   )      
   stoggle(
      "Ambulance Allocation",
      """[VI12188]Ambulance Allocation - Time series analysis of ambulance dispatches in the City of San Diego.""",
   )      
   stoggle(
      "Response Time",
      """[VI12189]Response Time - An analysis on the improvements of ambulance response time.""",
   )      
   stoggle(
      "Optimal Routing",
      """[VI12190]Optimal Routing - Project to find optimal routing of ambulances in Ithaca.""",
   )      
   stoggle(
      "Crash Analysis",
      """[VI12191]Crash Analysis - Predicting the probability of accidents on a given segment on a given time.""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="35")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def disaster():       
   stoggle(
      "Conflict Prediction",
      """[VI12192]Conflict Prediction - Notebooks on conflict prediction.""",
   )      
   stoggle(
      "Burglary Prediction",
      """[VI12193]Burglary Prediction - Spatio-Temporal Modelling for burglary prediction.""",
   )      
   stoggle(
      "Predicting Disease Outbreak ",
      """[VI12194]Predicting Disease Outbreak - Machine Learning implementation based on multiple classifier algorithm implementations.""",
   )      
   stoggle(
      "Road accident prediction",
      """[VI12195]Road accident prediction - Prediction on type of victims on federal road accidents in Brazil.""",
   )      
   stoggle(
      "Text Mining",
      """[VI12196]Text Mining - Disaster Management using Text mining.""",
   )      
   stoggle(
      "Twitter and disasters",
      """[VI12197]Twitter and disasters - Try to correctly predict whether tweets that are about disasters.""",
   )      
   stoggle(
      "Flood Risk",
      """[VI12198]Flood Risk - Impact of catastrophic flood events.""",
   )      
   stoggle(
      "Fire Prediction",
      """[VI12199]Fire Prediction - We used 4 different algorithms to predict the likelihood of future fires.""",
   )   
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="36")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def trading():      
   stoggle(
      "Deep Portfolio",
      """[VI12200]Deep Portfolio - Deep learning for finance Predict volume of bonds.""",
   )      
   stoggle(
      "AI Trading",
      """[VI12201]AI Trading - Modern AI trading techniques.""",
   )      
   stoggle(
      "Corporate Bonds",
      """[VI12202]Corporate Bonds - Predicting the buying and selling volume of the corporate bonds.""",
   )      
   stoggle(
      "Simulation",
      """[VI12203]Simulation - Investigating simulations as part of computational finance.""",
   )      
   stoggle(
      "Industry Clustering",
      """[VI12204]Industry Clustering - Project to cluster industries according to financial attributes.""",
   )      
   stoggle(
      "Financial Modeling",
      """[VI12205]Financial Modeling - HFT trading and implied volatility modeling.""",
   )      
   stoggle(
      "Trend Following",
      """[VI12206]Trend Following - A futures trend following portfolio investment strategy.""",
   )      
   stoggle(
      "Financial Statement Sentiment",
      """[VI12207]Financial Statement Sentiment - Extracting sentiment from financial statements using neural networks.""",
   )      
   stoggle(
      "Applied Corporate Finance",
      """[VI12208]Applied Corporate Finance - Studies the empirical behaviors in stock market.""",
   )      
   stoggle(
      "Market Crash Prediction",
      """[VI12209]Market Crash Prediction - Predicting market crashes using an LPPL model.""",
   )      
   stoggle(
      "NLP Finance Papers",
      """[VI12210]NLP Finance Papers - Curating quantitative finance papers using machine learning.""",
   )      
   stoggle(
      "ARIMA-LTSM Hybrid",
      """[VI12211]ARIMA-LTSM Hybrid - Hybrid model to predict future price correlation coefficients of two assets""",
   )      
   stoggle(
      "Basic Investments",
      """[VI12212]Basic Investments - Basic investment tools in python.""",
   )      
   stoggle(
      "Basic Derivatives",
      """[VI12223]Basic Derivatives - Basic forward contracts and hedging.""",
   )      
   stoggle(
      "Basic Finance",
      """[VI12224]Basic Finance - Source code notebooks basic finance applications.""",
   )      
   stoggle(
      "Advanced Pricing ML",
      """[VI12225]Advanced Pricing ML - Additional implementation of Advances in Financial Machine Learning (Book)""",
   )      
   stoggle(
      "Options and Regression",
      """[VI12226]Options and Regression - Financial engineering project for option pricing techniques.""",
   )      
   stoggle(
      "Quant Notebooks",
      """[VI12227]Quant Notebooks - Educational notebooks on quant finance, algorithmic trading and investment strategy.""",
   )      
   stoggle(
      "Forecasting Challenge",
      """[VI12228]Forecasting Challenge - Financial forecasting challenge by G-Research (Hedge Fund)""",
   )      
   stoggle(
      "XGboost",
      """[VI12229]XGboost - A trading algorithm using XgBoost""",
   )      
   stoggle(
      "Research Paper Trading",
      """[VI12230]Research Paper Trading - A strategy implementation based on a paper using Alpaca Markets.""",
   )      
   stoggle(
      "Various",
      """[VI12231]Various - Options, Allocation, Simulation""",
   )      
   stoggle(
      "ML & RL NYU",
      """[VI12232]ML & RL NYU - Machine Learning and Reinforcement Learning in Finance.2""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="37")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy() 
   
def gen():        
   stoggle(
      "zEpid",
      """[VI12233]zEpid - Epidemiology analysis package.""",
   )      
   stoggle(
      "Python For Epidemiologists",
      """[VI12234]Python For Epidemiologists - Tutorial to introduce epidemiology analysis in Python.""",
   )      
   stoggle(
      "Prescription Compliance",
      """[VI12235]Prescription Compliance - An analysis of prescription and medical compliance""",
   )      
   stoggle(
      "Respiratory Disease",
      """[VI12236]Respiratory Disease - Tracking respiratory diseases in Olympic athletes""",
   )      
   stoggle(
      "Bubonic Plague",
      """[VI12237]Bubonic Plague - Bubonic plague and SIR model.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="38")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def tools():         
   stoggle(
      "LexPredict",
      """[VI12238]LexPredict - Software package and library.""",
   )      
   stoggle(
      "AI Para-legal",
      """[VI12239]AI Para-legal - Lobe is the world's first AI paralegal.""",
   )         
   stoggle(
      "Legal Entity Detection",
      """[VI12240]Legal Entity Detection - NER For Legal Documents.""",
   )         
   stoggle(
      "Legal Case Summarisation",
      """[VI12241]Legal Case Summarisation - Implementation of different summarisation algorithms applied to legal case judgements.""",
   )         
   stoggle(
      "Legal Documents Google Scholar",
      """[VI12242]Legal Documents Google Scholar - Using Google scholar to extract cases programatically.""",
   )         
   stoggle(
      "Chat Bot",
      """[VI12243]Chat Bot - Chat-bot and email notifications.""",
   )         
   stoggle(
      "Congress API",
      """[VI12244]Congress API - ProPublica congress API access.""",
   )         
   stoggle(
      "Data Generator GDPR",
      """[VI12245]Data Generator GDPR - Dummy data generator for GDPR compliance""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="39")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def mgen():           
   stoggle(
      "Green Manufacturing",
      """[VI12246]Green Manufacturing - Mercedes-Benz Greener Manufacturing competition on Kaggle.""",
   )         
   stoggle(
      "Semiconductor Manufacturing",
      """[VI12247]Semiconductor Manufacturing - Semicondutor manufacturing process line data analysis.""",
   )         
   stoggle(
      "Smart Manufacturing",
      """[VI12248]Smart Manufacturing - Shared work of a modelling Methodology.""",
   )         
   stoggle(
      "Bosch Manufacturing",
      """[VI12249]Bosch Manufacturing - Bosch manufacturing project, Kaggle.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="40")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def mmain():           
   stoggle(
      "Predictive Maintenance 1",
      """[VI12250]Predictive Maintenance 1 - Predict remaining useful life of aircraft engines""",
   )         
   stoggle(
      "Predictive Maintenance 2",
      """[VI12251]Predictive Maintenance 2 - Time-To-Failure (TTF) or Remaining Useful Life (RUL)""",
   )         
   stoggle(
      "Manufacturing Maintenance",
      """[VI12252]Manufacturing Maintenance - Simulation of maintenance in manufacturing systems.""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="41")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy() 
   
def failure():         
   stoggle(
      "Predictive Analytics",
      """[VI12253]Predictive Analytics - Method for Predicting failures in Equipment using Sensor data.""",
   )         
   stoggle(
      "Detecting Defects",
      """[VI12254]Detecting Defects - Anomaly detection for defective semiconductors""",
   )         
   stoggle(
      "Defect Detection",
      """[VI12255]Defect Detection - Smart defect detection for pill manufacturing.""",
   )         
   stoggle(
      "Manufacturing Failures",
      """[VI12256]Manufacturing Failures - Reducing manufacturing failures.""",
   )         
   stoggle(
      "Manufacturing Anomalies",
      """[VI12257]Manufacturing Anomalies - Intelligent anomaly detection for manufacturing line.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="42")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def quality():           
   stoggle(
      "Quality Control",
      """[VI12258]Quality Control - Bosh failure of quality control.""",
   )         
   stoggle(
      "Manufacturing Quality",
      """[VI12259]Manufacturing Quality - Intelligent Manufacturing Quality Forecast""",
   )         
   stoggle(
      "Auto Manufacturing",
      """[VI12260]Auto Manufacturing - Regression Case Study Project on Manufacturing Auction Sale Data.""",
   )
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="43")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def marketing():            
   stoggle(
      "Video Popularity",
      """[VI12261]Video Popularity - HIP model for predicting the popularity of videos.""",
   )         
   stoggle(
      "YouTube transcriber",
      """[VI12262]YouTube transcriber - Automatically transcribe YouTube videos.""",
   )         
   stoggle(
      "Marketing Analytics",
      """[VI12263]Marketing Analytics - Marketing analytics case studies.""",
   )         
   stoggle(
      "Algorithmic Marketing",
      """[VI12264]Algorithmic Marketing - Models from Introduction to Algorithmic Marketing book""",
   )         
   stoggle(
      "Marketing Scripts",
      """[VI12265]Marketing Scripts - Marketing data science applications.""",
   )         
   stoggle(
      "Social Mining",
      """[VI12266]Social Mining - Mining the social web.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="44")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()   
   
def art():        
   stoggle(
      "Painting Forensics",
      """[VI12267]Painting Forensics - Analysing paintings to find out their year of creation.""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="45")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy() 
   
def tourism():         
   stoggle(
      "Flickr",
      """[VI12268]Flickr - Metadata mining tool for tourism research.""",
   )         
   stoggle(
      "Fashion",
      """[VI12269]Fashion - A clothing retrieval and visual recommendation model for fashion images""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="46")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
  
def elect():        
   stoggle(
      "Electricity Price",
      """[VI12270]Electricity Price - Electricity price comparison Singapore.""",
   )         
   stoggle(
      "Electricity-Coal Correlation",
      """[VI12271]Electricity-Coal Correlation - Determining the correlation between state electricity rates and coal generation over the past decade.""",
   )         
   stoggle(
      "Electricity Capacity",
      """[VI12272]Electricity Capacity - A Los Angeles Times analysis of California's costly power glut.""",
   )         
   stoggle(
      "Electricity Systems",
      """[VI12273]Electricity Systems - Optimal Wind+Hydrogen+Other+Battery+Solar (WHOBS) electricity systems for European countries.""",
   )         
   stoggle(
      "Load Disaggregation",
      """[VI12274]Load Disaggregation - Smart meter load disaggregation with Hidden Markov Models""",
   )         
   stoggle(
      "Price Forecasting",
      """[VI12275]Price Forecasting - Forecasting Day-Ahead electricity prices in the German bidding zone with deep neural networks.""",
   )         
   stoggle(
      "Carbon Index",
      """[VI12276]Carbon Index - Calculation of electricity CO₂ intensity at national, state, and NERC regions from 2001-present.""",
   )         
   stoggle(
      "Demand Forecasting",
      """[VI12277]Demand Forecasting - Electricity demand forecasting for Austin.""",
   )         
   stoggle(
      "Electricity Consumption",
      """[VI12278]Electricity Consumption - Estimating Electricity Consumption from Household Surveys""",
   )         
   stoggle(
      "Household power consumption",
      """[VI12279]Household power consumption - Individual household power consumption LSTM.""",
   )         
   stoggle(
      "Electricity French Distribution",
      """[VI12280]Electricity French Distribution - An analysis of electricity data provided by the French Distribution Network (RTE)""",
   )         
   stoggle(
      "Renewable Power Plants",
      """[VI12281]Renewable Power Plants - Time series of cumulated installed capacity.""",
   )         
   stoggle(
      "Wind Farm Flow",
      """[VI12282]Wind Farm Flow - A repository of wind plant flow models connected to FUSED-Wind.""",
   )         
   stoggle(
      "Power Plant",
      """[VI12283]Power Plant - The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011).""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="47")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def coal():           
   stoggle(
      "Coal Phase Out",
      """[VI12284]Coal Phase Out - Generation adequacy issues with Germany’s coal phaseout.""",
   )         
   stoggle(
      "Coal Prediction",
      """[VI12285]Coal Prediction - Predicting coal production.""",
   )         
   stoggle(
      "Oil & Gas",
      """[VI12286]Oil & Gas - Oil & Natural Gas price prediction using ARIMA & Neural Networks""",
   )         
   stoggle(
      "Gas Formula",
      """[VI12287]Gas Formula - Calculating potential economic effect of price indexation formula.""",
   )         
   stoggle(
      "Demand Prediction",
      """[VI12288]Demand Prediction - Natural gas demand prediction.""",
   )         
   stoggle(
      "Consumption Forecasting",
      """[VI12289]Consumption Forecasting - Natural gas consumption forecasting.""",
   )         
   stoggle(
      "Gas Trade ",
      """[VI12290]Gas Trade - World Model for Natural Gas Trade.""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="48")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def water():          
   stoggle(
      "Safe Water",
      """[VI12291]Safe Water - Predict health-based drinking water violations in the United States.""",
   )         
   stoggle(
      "Hydrology Data",
      """[VI12292]Hydrology Data - A suite of convenience functions for exploring water data in Python.""",
   )         
   stoggle(
      "Water Observatory",
      """[VI12293]Water Observatory - Monitoring water levels of lakes and reservoirs using satellite imagery.""",
   )         
   stoggle(
      "Water Pipelines",
      """[VI12294]Water Pipelines - Using machine learning to find water pipelines in aerial images.""",
   )         
   stoggle(
      "Water Modelling",
      """[VI12295]Water Modelling - Australian Water Resource Assessment (AWRA) Community Modelling System.""",
   )         
   stoggle(
      "Drought Restrictions",
      """[VI12296]Drought Restrictions - A Los Angeles Times analysis of water usage after the state eased drought restrictions""",
   )         
   stoggle(
      "Flood Prediction",
      """[VI12297]Flood Prediction - Applying LSTM on river water level data""",
   )         
   stoggle(
      "Sewage Overflow",
      """[VI12298]Sewage Overflow - Insights into the sanitary sewage overflow (SSO).""",
   )         
   stoggle(
      "Water Accounting",
      """[VI12299]Water Accounting - Assembles water budget data for the US from existing data source""",
   )         
   stoggle(
      "Air Quality Prediction",
      """[VI12300]Air Quality Prediction - Predict air quality(aq) in Beijing and London in the next 48 hours.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="49")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()
   
def transportation():           
   stoggle(
      "Transdim",
      """[VI12301]Transdim - Creating accurate and efficient solutions for the spatio-temporal traffic data imputation and prediction tasks.""",
   )         
   stoggle(
      "Transport Recommendation",
      """[VI12302]Transport Recommendation - Context-Aware Multi-Modal Transportation Recommendation""",
   )         
   stoggle(
      "Transport Data",
      """[VI12303]Transport Data - Data and notebooks for Toronto transport.""",
   )         
   stoggle(
      "Transport Demand",
      """[VI12304]Transport Demand - Predicting demand for public transportation in Nairobi.""",
   )         
   stoggle(
      "Demand Estimation",
      """[VI12305]Demand Estimation - Implementation of dynamic origin-destination demand estimation.""",
   )         
   stoggle(
      "Congestion Analysis",
      """[VI12306]Congestion Analysis - Transportation systems analysis""",
   )         
   stoggle(
      "TS Analysis",
      """[VI12307]TS Analysis - Time series analysis on transportation data.""",
   )         
   stoggle(
      "Network Graph Subway",
      """[VI12308]Network Graph Subway - Vulnerability analysis for transportation networks.""",
   )         
   stoggle(
      "Transportation Inefficiencies",
      """[VI12309]Transportation Inefficiencies - Quantifying the inefficiencies of Transportation Networks""",
   )         
   stoggle(
      "Train Optimisation",
      """[VI12310]Train Optimisation - Train schedule optimisation""",
   )         
   stoggle(
      "Traffic Prediction",
      """[VI12311]Traffic Prediction - multi attention recurrent neural networks for time-series (city traffic)""",
   )         
   stoggle(
      "Predict Crashes",
      """[VI12312]Predict Crashes - Crash prediction modelling application that leverages multiple data sources""",
   )         
   stoggle(
      "AI Supply chain",
      """[VI12313]AI Supply chain - Supply chain optimisation system.""",
   )         
   stoggle(
      "Transfer Learning Flight Delay",
      """[VI12314]Transfer Learning Flight Delay - Using variation encoders in Keras to predict flight delay.""",
   )         
   stoggle(
      "Replenishment",
      """[VI12315]Replenishment - Retail replenishment code for supply chain management.""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="50")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy() 
   
def wholesale():         
   stoggle(
      "Customer Analysis",
      """[VI12316]Customer Analysis - Wholesale customer analysis.""",
   )         
   stoggle(
      "Distribution",
      """[VI12317]Distribution - JB wholesale distribution analysis.""",
   )         
   stoggle(
      "Clustering",
      """[VI12318]Clustering - Unsupervised learning techniques are applied on product spending data collected for customers""",
   )         
   stoggle(
      "Market Basket Analysis",
      """[VI12319]Market Basket Analysis - Instacart public dataset to report which products are often shopped together.""",
   )  
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUAL/GROUP_LEADER","GROUP"],
        value=('NONE'),key="51")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()       
   
def retail():   
   stoggle(
      "Retail Analysis",
      """[VI12320]Retail Analysis - Studying Online Retail Dataset and getting insights from it.""",
   )         
   stoggle(
      "Online Insights",
      """[VI12321]Online Insights - Analyzing the Online Transactions in UK""",
   )         
   stoggle(
      "Retail Use-case",
      """[VI12322]Retail Use-case - Notebooks & Data for CyberShop Retail Use Case""",
   )         
   stoggle(
      "Dwell Time",
      """[VI12323]Dwell Time - Customer dwell time and other analysis.""",
   )         
   stoggle(
      "Retail Cohort",
      """[VI12324]Retail Cohort - Cohort analysis.""",
   ) 
   add_vertical_space(1)
   start_color = st.select_slider(
       'Select INDIVIDUAL/GROUP_LEADER or Group',
        options=["NONE","INDIVIDUA/GROUP_LEADER","GROUP"],
        value=('NONE'),key="52")
   st.write('You selected :', start_color)
   if start_color=="INDIVIDUAL/GROUP_LEADER":
       rpey()
   if start_color=="GROUP":
       rpuy()        

if selected =="Info":
    st.title(f"{selected}")
    mdlit(
    """🎯 VINTERN is a platform that offers virtual internship opportunities for students. In order to meet the client's needs, we offer real-time issue statements. Based on their interests and available time, the students can work on these issue statements. They will be put on a shortlist for hiring based on their performance.
    
""",
)
    mdlit(
    """🎯 VINTERN provides 5 Intern packs. Students can choose any of the intern pack based on their available time.
"""
)
    mdlit(
    """🎯 At present VINTERN has 350+ live project ideas. Student can select any idea among these. Intern pack selection and project idea selection will be followed by a payment by the user. 
"""
)
    mdlit(
    """🎯After successfull payment by the student. The enrollment confirmation and Internship offer letter will be provided within 6-8 hours.
"""
)    
    mdlit(
    """🎯In case of group projects the group leader should first purchase the intern pack by choosing group pack. Group leader should invite his team mates by sharing his GROUP ID. The team mates should join the same project idea.
"""
) 
    mdlit(
    """🎯Daily attendance button to be toggled. Minimum 75% attendance is necessary. Every 15 days progress report doc shoould be uploaded.
"""
)      
    mdlit(
    """🎯Student can interact with mentor in [whatsapp](https://wa.me/message/NZAFYRRGHO6QP1) or uisng mail id: support@vintern.tech.
"""
)  
    mdlit(
    """🎯Every 15 day progress report submission is mandatory. All submissions will be considered. Dashboard will show importnat alerts if you miss any deadlines.
"""
)          
    mdlit(
    """🎯If the project is fully satisfied by client and VINTERN.tech team. Then you will be provided a hard & soft copy certificate with GOLDEN STAR.
"""
)       
    mdlit(
    """🎯If the project is averagely satisfied by client and VINTERN.tech team. Then you will be provided a soft copy certificate with SILVER STAR.
"""
)          
    mdlit(
    """🎯If we found skilled students in this entire journey they will be recruited in our VINTERN.tech. Groups which completely satisfied the project idea that group will be forwarded to stipend based remote internships with client.
"""
)     
    mdlit(
    """🎯For any queries/ demands feel free to contact us.
"""
)       
    
if selected =="Projects":
    st.title(f"{selected}")
    st.caption("Click on the drop down🔻 and copy the [VICODE] for enrolling.")
    tab01,tab02,tab03=st.tabs(["<| 0 |","| 1 |","| 2 |>"])
    with tab01:    
      tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs(["Accommodation & Food", "Accounting", "Agriculture", "Banking & Insurance", "Biotechnological & Life Sciences","Education & Research","Emergency & Police"])

      with tab1:
         tab101,tab102=st.tabs(["Food",'Restaurant'])
         with tab101:
            food()
         with tab102:
            restaurant()   

      with tab2:
         tab203,tab204,tab205,tab206=st.tabs(["Machine Learning",'Analytics','Textual Analysis','Data, Parsing and APIs'])
         with tab203:
            ml()
         with tab204:
            analytics()  
         with tab205:
            Textual_analytics()
         with tab206:
            parse()

      with tab3:
         tab307,tab308=st.tabs(["Economics","Development"])
         with tab307:
            economics()
         with tab308:
            development()

      with tab4:
         tab409,tab410,tab411,tab412,tab413,tab414,tab415=st.tabs(["Consumer Finance","Management and Operation","Valuation","Fraud","Insurance and Risk","Anomaly","Physical"])         
         with tab409:
            consumer()
         with tab410:
            management()
         with tab411:
            valuation()
         with tab412:
            fraud()
         with tab413:
            risk()
         with tab414:
            anomaly()
         with tab415:
            physical()
      with tab5:
         tab516,tab517,tab518,tab519=st.tabs(["General","Sequencing","Genomics","Life-sciences"])                     
         with tab516:
            general()
         with tab517:
            sequencing()
         with tab518:
            genomics()
         with tab519:
            life_sciences()      
      with tab6:
         tab620,tab621=st.tabs(["Student","School"])
         with tab620:
            student()
         with tab621:
            school()   
      with tab7:
         tab722,tab723,tab724,tab725=st.tabs(["Preventative and Reactive","Crime","Ambulance","Disaster Management"]) 
         with tab722:
            preventive()
         with tab723:
            crime()
         with tab724:
            ambulance()
         with tab725:
            disaster()             
    with tab02:
      tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15 = st.tabs(["Finance","Healthcare","Justics, Law & Regulations","Manufacturing","Media & Publishing","Utilities","Wholesale & Retail","Miscellaneous"])
      with tab8:
         tab823,tab824=st.tabs(["Trading and Investment","-"])
         with tab823:
            trading()
      with tab9:
         tab925,tab926=st.tabs(["General","-"])
         with tab925:
            gen()      
      with tab10:
         tab1001,tab1002=st.tabs(["Tools","-"])
         with tab1001:
            tools()
            
      with tab11:
         tab1103,tab1104,tab1105,tab1106=st.tabs(["General","Maintenance","Failure","Quality"])        
         with tab1103:
            mgen()
         with tab1104:
            mmain()
         with tab1105:
            failure()
         with tab1106:
            quality()    
      with tab12:
         tab1207,tab1208=st.tabs(["Marketing","-"])     
         with tab1207:
            marketing()       
      with tab13:
         tab1309,tab1310,tab1311,tab1312=st.tabs(["Electricity","Coal, Oil & Gas","Water & Pollution","Transportation"]) 
         with tab1309:
            elect()
         with tab1310:
            coal()
         with tab1311:
            water()
         with tab1312:
            transportation()
      with tab14:
         tab1413,tab1414=st.tabs(["Wholesale","Retail"])
         with tab1413:
            wholesale()
         with tab1414:
            retail()
      with tab15:
         tab1515,tab1516=st.tabs(["Art","Tourism"])
         with tab1515:
            art()
         with tab1516:
            tourism()
      
if selected =="Dashboard":
       names=["vatsavai vinay varma","Shvk ashok","dishni divya","Ramesh","Bhargav","Madhuram","Harshath","Chinteti Hari Krishna","HEMANTH","Devadasu","Nammi Divya Deepika","D. Vineel Kumar","D. Sandeep Kumar","Durga Prasad","G. Kailash Kumar","jayaram","haribabu","Harsha Vardhan","Sk.Afsana","Jayant Kumar","katakamsandeep","sandeepkatakam","K. Shiva Swamy","vasu","lakshmanbattula","Manikanta","R.Manivaran","NihanthNaidu007","Nakka Srinivas","Prahaladh2001","VENKATA PRASAD MERUGU ","R.Manivaran","Routhu Shanmukh","Sahithi Nalla","Botta Venkata Sai Gowri Prasanth","SamFreddie77","SANKAR ","R.Sreya","Surya Trinadh ","Tirumala","T. Gnyana Prasuna","Reddy upendra","Gaddi Vamshi","V.Sriharshavardhan","Govardhan ganesh ","V.Sai Praneeth","Vavilapalli Vineetha","Vineeth Killamsetty","G.Ganesh","prasanth  gupta","Pilla Manindhar"]
       usernames=["201801130001@cutmap.ac.in","201801330053@cutmap.ac.in","201801330054@cutmap.ac.in","bathularamesh13@gmail.com","bhargavsaivardhan@gmail.com","boyapatiram08@gmail.com","cheedellaharshath4891@gmail.com","chintetiharikrishna@gmail.com","chowdaryhemanth66@gmail.com","devadasb733@gmail.com","divyadeepika8978@gmail.com","doddavineel@gmail.com","dsandy5252@gmail.com","durgaprasadmolleti5@gmail.com","gandepalli5089@gmail.com","geddajayaram4@gmail.com","haripuvvada55@gmail.com","harshavasu6699@gmail.com","honeyafsana5@gmail.com","jayantchowtha143@gmail.com","katakamsandeep111@gmail.com","katakamsandeep6976@gmail.com","konashiva5@gmail.com","kuriminellivasu@gmail.com","lakshmanbattula@gmail.com","manipuppala9985@gmail.com","manivaranronanki@gmail.com","nihanthnaidu007@gmail.com","nivastech90@gmail.com","prahaladh8074995565@gmail.com","prasadmv6305@gmail.com","ronakimanivaran@gmail.com","routhushanmukh2002@gmail.com","sahithinalla2304@gmail.com","saigowriprasanth@gmail.com","samfreddie77@gmail.com","sankerramireddy@gmail.com","sreyaramchandran961@gmail.com","suryasrivalli877@gmail.com","tirumalakumarbehara@gmail.com","tirumellagnyanaprasuna@gmail.com","upendra28062003@gmail.com","vamshiyadav961@gmail.com","vardhanharsha00384@gmail.com","vechalapugovardhan2002@gmail.com","vemulapallisaipraneeth@gmail.com","vineetha1022@gmail.com","vineethkillamsetty8@gmail.com","201801330037@cutmap.ac.in","201801330022@cutmap.ac.in","manindharpilla@gmail.com"]

       file_path=Path(__file__).parent / "hashed_pw.pkl"
       with file_path.open("rb") as file:
           hashed_passwords=pickle.load(file)
    
       authenticator=stauth.Authenticate(names, usernames, hashed_passwords, "VINTERN", "VINTERN", cookie_expiry_days=30)
       name, authentication_status, username = authenticator.login("Login","main")
        
       if authentication_status == False:
          st.error("Email/password is incorrect")
          st.caption("Your email id is your Username")  
       if authentication_status == None:
          st.warning("Please enter your email and password")
          st.caption("Your email id is your Username")  
       if authentication_status:
          st.caption(f"Welcome {name}")
          if username=="routhushanmukh2002@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP1]")
            st.info("[VI12004] Food Classification")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Routhu Shanmukh : routhushanmukh2002@gmail.com")
                st.write("Prasanth Botta : saigowriprasanth@gmail.com")
                st.write("Reddy Upendra : upendra28062003@gmail.com")
                st.write("Nakka Srinivas : nivastech90@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Since food is essential to life since it gives us various nutrients, it is important for each person to keep a close eye on their eating patterns. Food categorization is therefore vital for living a better lifestyle. Pre-trained models to be employed in this project instead of the more conventional ways of creating a model from scratch, which reduces computing time and costs while also producing superior results.
            For training and validating, a food dataset with several classes and numerous photos within each class to be collected. Develop some pre-trained algorithms which will identify the provided food and make nutritional content predictions based on the image's colour. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz you can use this data. You can also add some more images.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1AgeoNjfoKT4c2YaxAUeEOFsAmuaXcBCy?usp=sharing)""")
          
          if username=="saigowriprasanth@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP1]")
            st.info("[VI12004] Food Classification")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Routhu Shanmukh : routhushanmukh2002@gmail.com")
                st.write("Prasanth Botta : saigowriprasanth@gmail.com")
                st.write("Reddy Upendra : upendra28062003@gmail.com")
                st.write("Nakka Srinivas : nivastech90@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Since food is essential to life since it gives us various nutrients, it is important for each person to keep a close eye on their eating patterns. Food categorization is therefore vital for living a better lifestyle. Pre-trained models to be employed in this project instead of the more conventional ways of creating a model from scratch, which reduces computing time and costs while also producing superior results.
            For training and validating, a food dataset with several classes and numerous photos within each class to be collected. Develop some pre-trained algorithms which will identify the provided food and make nutritional content predictions based on the image's colour. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz you can use this data. You can also add some more images.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1AgeoNjfoKT4c2YaxAUeEOFsAmuaXcBCy?usp=sharing)""")
          
          if username=="upendra28062003@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP1]")
            st.info("[VI12004] Food Classification")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Routhu Shanmukh : routhushanmukh2002@gmail.com")
                st.write("Prasanth Botta : saigowriprasanth@gmail.com")
                st.write("Reddy Upendra : upendra28062003@gmail.com")
                st.write("Nakka Srinivas : nivastech90@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Since food is essential to life since it gives us various nutrients, it is important for each person to keep a close eye on their eating patterns. Food categorization is therefore vital for living a better lifestyle. Pre-trained models to be employed in this project instead of the more conventional ways of creating a model from scratch, which reduces computing time and costs while also producing superior results.
            For training and validating, a food dataset with several classes and numerous photos within each class to be collected. Develop some pre-trained algorithms which will identify the provided food and make nutritional content predictions based on the image's colour. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz you can use this data. You can also add some more images.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1AgeoNjfoKT4c2YaxAUeEOFsAmuaXcBCy?usp=sharing)""")
          
          if username=="nivastech90@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP1]")
            st.info("[VI12004] Food Classification")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Routhu Shanmukh : routhushanmukh2002@gmail.com")
                st.write("Prasanth Botta : saigowriprasanth@gmail.com")
                st.write("Reddy Upendra : upendra28062003@gmail.com")
                st.write("Nakka Srinivas : nivastech90@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Since food is essential to life since it gives us various nutrients, it is important for each person to keep a close eye on their eating patterns. Food categorization is therefore vital for living a better lifestyle. Pre-trained models to be employed in this project instead of the more conventional ways of creating a model from scratch, which reduces computing time and costs while also producing superior results.
            For training and validating, a food dataset with several classes and numerous photos within each class to be collected. Develop some pre-trained algorithms which will identify the provided food and make nutritional content predictions based on the image's colour. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz you can use this data. You can also add some more images.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1AgeoNjfoKT4c2YaxAUeEOFsAmuaXcBCy?usp=sharing)""")
        
          if username=="lakshmanbattula@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP3]")
            
          if username=="gandepalli5089@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP6]") 
            st.info("[VI12087] Medical Insurance Claim")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("G. Kailash Kumar : gandepalli5089@gmail.com")
                st.write("Jayanth Kumar : jayantchowtha143@gmail.com")
                st.write("T. Gnyana Prasuna : tirumellagnyanaprasuna@gmail.com")
                st.write("D. Sandeep Kumar : dsandy5252@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Build a model to predict whether a person can claim his medical insurance or not. Focus on some factors / parameters that can help. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/sharmaroshan/Insurance-Claim-Prediction/blob/master/insurance.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/111HX8CsfQ2Oqi0q3An3kY4YqG1vEB6OV?usp=sharing)""")
          
          if username=="jayantchowtha143@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP6]") 
            st.info("[VI12087] Medical Insurance Claim")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("G. Kailash Kumar : gandepalli5089@gmail.com")
                st.write("Jayanth Kumar : jayantchowtha143@gmail.com")
                st.write("T. Gnyana Prasuna : tirumellagnyanaprasuna@gmail.com")
                st.write("D. Sandeep Kumar : dsandy5252@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Build a model to predict whether a person can claim his medical insurance or not. Focus on some factors / parameters that can help. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/sharmaroshan/Insurance-Claim-Prediction/blob/master/insurance.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/111HX8CsfQ2Oqi0q3An3kY4YqG1vEB6OV?usp=sharing)""")
                
          if username=="tirumellagnyanaprasuna@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP6]") 
            st.info("[VI12087] Medical Insurance Claim")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("G. Kailash Kumar : gandepalli5089@gmail.com")
                st.write("Jayanth Kumar : jayantchowtha143@gmail.com")
                st.write("T. Gnyana Prasuna : tirumellagnyanaprasuna@gmail.com")
                st.write("D. Sandeep Kumar : dsandy5252@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Build a model to predict whether a person can claim his medical insurance or not. Focus on some factors / parameters that can help. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/sharmaroshan/Insurance-Claim-Prediction/blob/master/insurance.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/111HX8CsfQ2Oqi0q3An3kY4YqG1vEB6OV?usp=sharing)""") 
                
          if username=="dsandy5252@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP6]") 
            st.info("[VI12087] Medical Insurance Claim")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("G. Kailash Kumar : gandepalli5089@gmail.com")
                st.write("Jayanth Kumar : jayantchowtha143@gmail.com")
                st.write("T. Gnyana Prasuna : tirumellagnyanaprasuna@gmail.com")
                st.write("D. Sandeep Kumar : dsandy5252@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Build a model to predict whether a person can claim his medical insurance or not. Focus on some factors / parameters that can help. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/sharmaroshan/Insurance-Claim-Prediction/blob/master/insurance.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/111HX8CsfQ2Oqi0q3An3kY4YqG1vEB6OV?usp=sharing)""")      
          
          if username=="201801130001@cutmap.ac.in":
            st.write("Dear user your GROUP INVITE is [GVIOP7]")
            st.info("[VI12241] Legal Case Summarization")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("vatsavai vinay varma : 201801130001@cutmap.ac.in")
                st.write("Nammi Divya Deepika : divyadeepika8978@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Legal Case Document Summarization involves Extractive and Abstractive Methods. Use NLP to summarize the legal dcomunets contents. You can use any text summarization approach. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset http://www.liiofindia.org/in/cases/cen/INSC/ you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1dcZYgip2vdhtcQUTT0Xmn2V4qpqo7wMN?usp=sharing)""")
          
          if username=="divyadeepika8978@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP7]")
            st.info("[VI12241] Legal Case Summarization")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("vatsavai vinay varma : 201801130001@cutmap.ac.in")
                st.write("Nammi Divya Deepika : divyadeepika8978@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Legal Case Document Summarization involves Extractive and Abstractive Methods. Use NLP to summarize the legal dcomunets contents. You can use any text summarization approach. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset http://www.liiofindia.org/in/cases/cen/INSC/ you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")        
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1dcZYgip2vdhtcQUTT0Xmn2V4qpqo7wMN?usp=sharing)""")
          
          if username=="manipuppala9985@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP8]")
            st.info("VI12081Used Car - Used vehicle price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Manikanta : manipuppala9985@gmail.com")
                st.write("Durga Prasad : durgaprasadmolleti5@gmail.com")
                st.write("Tirumala : tirumalakumarbehara@gmail.com")
                st.write("Ramesh : bathularamesh13@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model which can prdict the price of a used car. You should work with some parameters which affect car price and capacity. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")        
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rWyeL-ZOYyCWD-GZ0IsGhX2FynzuJhyP?usp=sharing)""")
                
          if username=="durgaprasadmolleti5@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP8]")
            st.info("VI12081Used Car - Used vehicle price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Manikanta : manipuppala9985@gmail.com")
                st.write("Durga Prasad : durgaprasadmolleti5@gmail.com")
                st.write("Tirumala : tirumalakumarbehara@gmail.com")
                st.write("Ramesh : bathularamesh13@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model which can prdict the price of a used car. You should work with some parameters which affect car price and capacity. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")        
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rWyeL-ZOYyCWD-GZ0IsGhX2FynzuJhyP?usp=sharing)""")
                
          if username=="tirumalakumarbehara@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP8]")
            st.info("VI12081Used Car - Used vehicle price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Manikanta : manipuppala9985@gmail.com")
                st.write("Durga Prasad : durgaprasadmolleti5@gmail.com")
                st.write("Tirumala : tirumalakumarbehara@gmail.com")
                st.write("Ramesh : bathularamesh13@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model which can prdict the price of a used car. You should work with some parameters which affect car price and capacity. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")        
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rWyeL-ZOYyCWD-GZ0IsGhX2FynzuJhyP?usp=sharing)""")
                
          if username=="bathularamesh13@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP8]")
            st.info("VI12081Used Car - Used vehicle price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Manikanta : manipuppala9985@gmail.com")
                st.write("Durga Prasad : durgaprasadmolleti5@gmail.com")
                st.write("Tirumala : tirumalakumarbehara@gmail.com")
                st.write("Ramesh : bathularamesh13@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model which can prdict the price of a used car. You should work with some parameters which affect car price and capacity. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")        
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 21,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rWyeL-ZOYyCWD-GZ0IsGhX2FynzuJhyP?usp=sharing)""")
                
          if username=="201801330053@cutmap.ac.in":
            st.write("Dear user your GROUP INVITE is [GVIOP9]")
            st.info("VI12245 Data Generator GDPR - Dummy data generator for GDPR compliance")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("shvk ashok : 201801330053@cutmap.ac.in")
                st.write("Dishni Divya : 201801330054@cutmap.ac.in")
                st.write("SamFreddie77 : samfreddie77@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a dummy data generator which can be use for GDPR compliance when testing on cloud services. It should be used for any scenario where data or a specific format is needed. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/toningega/Data_Generator/blob/master/Random_names_master.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/13grVFW-8xHPRj89713HHcB7J1TTbIQ66?usp=sharing)""")  
            
          if username=="201801330054@cutmap.ac.in":
            st.write("Dear user your GROUP INVITE is [GVIOP9]")
            st.info("VI12245 Data Generator GDPR - Dummy data generator for GDPR compliance")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("shvk ashok : 201801330053@cutmap.ac.in")
                st.write("Dishni Divya : 201801330054@cutmap.ac.in")
                st.write("SamFreddie77 : samfreddie77@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a dummy data generator which can be use for GDPR compliance when testing on cloud services. It should be used for any scenario where data or a specific format is needed. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/toningega/Data_Generator/blob/master/Random_names_master.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")     
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/13grVFW-8xHPRj89713HHcB7J1TTbIQ66?usp=sharing)""")  
              
          if username=="samfreddie77@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP9]")
            st.info("VI12245 Data Generator GDPR - Dummy data generator for GDPR compliance")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("shvk ashok : 201801330053@cutmap.ac.in")
                st.write("Dishni Divya : 201801330054@cutmap.ac.in")
                st.write("SamFreddie77 : samfreddie77@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a dummy data generator which can be use for GDPR compliance when testing on cloud services. It should be used for any scenario where data or a specific format is needed. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/toningega/Data_Generator/blob/master/Random_names_master.csv you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Manoj")     
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
                
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/13grVFW-8xHPRj89713HHcB7J1TTbIQ66?usp=sharing)""")  
              
          if username=="vamshiyadav961@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP11]")  
            
          if username=="honeyafsana5@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP12]")
            st.info("VI12007 Fine Food Reviews - Sentiment analysis on Amazon Fine Food Reviews.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Sk.Afsana : honeyafsana5@gmail.com")
                st.write("V.Sriharshavardhan : vardhanharsha00384@gmail.com")
                st.write("R.Sreya : sreyaramchandran961@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a sentiment analysis system for amazon fine food reviews. The purpose of this analysis is to make up a prediction model where you should be able to predict whether a recommendation is positive or negative. In this analysis, you should not focus on the Score, but only the positive/negative sentiment of the recommendation. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://www.kaggle.com/snap/amazon-fine-food-reviews you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Bhuvi")     
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1CYouJQMiooBJVZt_eHL14aGtKT3q72lZ?usp=sharing)""")
          
          if username=="vardhanharsha00384@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP12]")
            st.info("VI12007 Fine Food Reviews - Sentiment analysis on Amazon Fine Food Reviews.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Sk.Afsana : honeyafsana5@gmail.com")
                st.write("V.Sriharshavardhan : vardhanharsha00384@gmail.com")
                st.write("R.Sreya : sreyaramchandran961@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Create a sentiment analysis system for amazon fine food reviews. The purpose of this analysis is to make up a prediction model where you should be able to predict whether a recommendation is positive or negative. In this analysis, you should not focus on the Score, but only the positive/negative sentiment of the recommendation. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://www.kaggle.com/snap/amazon-fine-food-reviews you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Bhuvi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1CYouJQMiooBJVZt_eHL14aGtKT3q72lZ?usp=sharing)""")
          
          
          if username=="sreyaramchandran961@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP12]")
            st.info("VI12007 Fine Food Reviews - Sentiment analysis on Amazon Fine Food Reviews.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Sk.Afsana : honeyafsana5@gmail.com")
                st.write("V.Sriharshavardhan : vardhanharsha00384@gmail.com")
                st.write("R.Sreya : sreyaramchandran961@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Create a sentiment analysis system for amazon fine food reviews. The purpose of this analysis is to make up a prediction model where you should be able to predict whether a recommendation is positive or negative. In this analysis, you should not focus on the Score, but only the positive/negative sentiment of the recommendation. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://www.kaggle.com/snap/amazon-fine-food-reviews you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Bhuvi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1CYouJQMiooBJVZt_eHL14aGtKT3q72lZ?usp=sharing)""")
        
          
          
          if username=="devadasb733@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP13]")
            st.info("VI12059 Segmentation - Agricultural field parcel segmentation using satellite images.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Devadasu : devadasb733@gmail.com")
                st.write("Harshath : cheedellaharshath4891@gmail.com")
                st.write("haribabu : haripuvvada55@gmail.com")
                
            st.title("Problem Statement:")
            st.write(""" Aim to delineate agricultural field parcels from satellite images via deep learning instance segmentation. Manual delineation is accurate but time consuming, and many automated approaches with traditional image segmentation fields quickly using deep learning concepts. techniques struggle to capture the variety of possible field appearances. So create a model that can segment  Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/chrieke/InstanceSegmentation_Sentinel2/tree/master/data you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Ranjeeth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rRYvMvrsxsnfJb7Y80mDDxWnRkc7kekW?usp=sharing)""") 
          
          if username=="cheedellaharshath4891@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP13]")
            st.info("VI12059 Segmentation - Agricultural field parcel segmentation using satellite images.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Devadasu : devadasb733@gmail.com")
                st.write("Harshath : cheedellaharshath4891@gmail.com")
                st.write("haribabu : haripuvvada55@gmail.com")
                
            st.title("Problem Statement:")
            st.write(""" Aim to delineate agricultural field parcels from satellite images via deep learning instance segmentation. Manual delineation is accurate but time consuming, and many automated approaches with traditional image segmentation fields quickly using deep learning concepts. techniques struggle to capture the variety of possible field appearances. So create a model that can segment  Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/chrieke/InstanceSegmentation_Sentinel2/tree/master/data you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Ranjeeth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("9/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rRYvMvrsxsnfJb7Y80mDDxWnRkc7kekW?usp=sharing)""")
                
          if username=="haripuvvada55@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP13]")
            st.info("VI12059 Segmentation - Agricultural field parcel segmentation using satellite images.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Devadasu : devadasb733@gmail.com")
                st.write("Harshath : cheedellaharshath4891@gmail.com")
                st.write("haribabu : haripuvvada55@gmail.com")
                
            st.title("Problem Statement:")
            st.write(""" Aim to delineate agricultural field parcels from satellite images via deep learning instance segmentation. Manual delineation is accurate but time consuming, and many automated approaches with traditional image segmentation fields quickly using deep learning concepts. techniques struggle to capture the variety of possible field appearances. So create a model that can segment  Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/chrieke/InstanceSegmentation_Sentinel2/tree/master/data you can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Ranjeeth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1rRYvMvrsxsnfJb7Y80mDDxWnRkc7kekW?usp=sharing)""")
             
          if username=="katakamsandeep6976@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP14]")
            st.info("[VI12054] Prices - Agricultural price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("katakamsandeep : katakamsandeep6976@gmail.com & katakamsandeep111@gmail.com")
                st.write("Bhargav : bhargavsaivardhan@gmail.com")
                st.write("NihanthNaidu007 : nihanthnaidu007@gmail.com")
                st.write("Vineeth Killamsetty : vineethkillamsetty8@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Agricultural price prediction : Build a model that will predict price via market wise and commodity wise. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/wholesale_price.csv  &  https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/total.csv . You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Pranavi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1k75VRKBlEa0_JjV6_Xp2Ol99qZNfAy9D?usp=sharing)""")
                
          if username=="bhargavsaivardhan@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP14]")
            st.info("[VI12054] Prices - Agricultural price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("katakamsandeep : katakamsandeep6976@gmail.com & katakamsandeep111@gmail.com")
                st.write("Bhargav : bhargavsaivardhan@gmail.com")
                st.write("NihanthNaidu007 : nihanthnaidu007@gmail.com")
                st.write("Vineeth Killamsetty : vineethkillamsetty8@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Agricultural price prediction : Build a model that will predict price via market wise and commodity wise. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/wholesale_price.csv  &  https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/total.csv . You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Pranavi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1k75VRKBlEa0_JjV6_Xp2Ol99qZNfAy9D?usp=sharing)""")
                
          if username=="nihanthnaidu007@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP14]")
            st.info("[VI12054] Prices - Agricultural price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("katakamsandeep : katakamsandeep6976@gmail.com & katakamsandeep111@gmail.com")
                st.write("Bhargav : bhargavsaivardhan@gmail.com")
                st.write("NihanthNaidu007 : nihanthnaidu007@gmail.com")
                st.write("Vineeth Killamsetty : vineethkillamsetty8@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Agricultural price prediction : Build a model that will predict price via market wise and commodity wise. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/wholesale_price.csv  &  https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/total.csv . You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Pranavi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1k75VRKBlEa0_JjV6_Xp2Ol99qZNfAy9D?usp=sharing)""")
                
          if username=="vineethkillamsetty8@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP14]")
            st.info("[VI12054] Prices - Agricultural price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("katakamsandeep : katakamsandeep6976@gmail.com & katakamsandeep111@gmail.com")
                st.write("Bhargav : bhargavsaivardhan@gmail.com")
                st.write("NihanthNaidu007 : nihanthnaidu007@gmail.com")
                st.write("Vineeth Killamsetty : vineethkillamsetty8@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Agricultural price prediction : Build a model that will predict price via market wise and commodity wise. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/wholesale_price.csv  &  https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/total.csv . You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Pranavi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1k75VRKBlEa0_JjV6_Xp2Ol99qZNfAy9D?usp=sharing)""")     
                
          if username=="katakamsandeep111@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP14]")
            st.info("[VI12054] Prices - Agricultural price prediction")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("katakamsandeep : katakamsandeep6976@gmail.com & katakamsandeep111@gmail.com")
                st.write("Bhargav : bhargavsaivardhan@gmail.com")
                st.write("NihanthNaidu007 : nihanthnaidu007@gmail.com")
                st.write("Vineeth Killamsetty : vineethkillamsetty8@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Agricultural price prediction : Build a model that will predict price via market wise and commodity wise. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/wholesale_price.csv  &  https://github.com/deadskull7/Agricultural-Price-Prediction-and-Visualization-on-Android-App/blob/master/total.csv . You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Pranavi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1k75VRKBlEa0_JjV6_Xp2Ol99qZNfAy9D?usp=sharing)""")      
          
          if username=="boyapatiram08@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP15]")
            st.info("VI12132 Leaf Identification - Identification of plants through plant leaves on the basis of their shape, color and texture.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Madhuram: boyapatiram08@gmail.com")
                st.write("Chinteti Hari Krishna : chintetiharikrishna@gmail.com")
                st.write("SANKAR  : sankerramireddy@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model that can identify plants through plant leaves on the basis of their shape, color and texture features using digital image processing techniques. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset  http://flavia.sourceforge.net/ You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Ranjeeth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1miKUYt-e3uJAJx2DNH9q1bKgCUl1xBQj?usp=sharing)""")
          
          if username=="chintetiharikrishna@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP15]")
            st.info("VI12132 Leaf Identification - Identification of plants through plant leaves on the basis of their shape, color and texture.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Madhuram: boyapatiram08@gmail.com")
                st.write("Chinteti Hari Krishna : chintetiharikrishna@gmail.com")
                st.write("SANKAR  : sankerramireddy@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Create a model that can identify plants through plant leaves on the basis of their shape, color and texture features using digital image processing techniques. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset  http://flavia.sourceforge.net/ You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Ranjeeth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1miKUYt-e3uJAJx2DNH9q1bKgCUl1xBQj?usp=sharing)""")
          
          if username=="sankerramireddy@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP15]")
            st.info("VI12132 Leaf Identification - Identification of plants through plant leaves on the basis of their shape, color and texture.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("Madhuram: boyapatiram08@gmail.com")
                st.write("Chinteti Hari Krishna : chintetiharikrishna@gmail.com")
                st.write("SANKAR  : sankerramireddy@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""Create a model that can identify plants through plant leaves on the basis of their shape, color and texture features using digital image processing techniques. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset  http://flavia.sourceforge.net/ You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Ranjeeth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1miKUYt-e3uJAJx2DNH9q1bKgCUl1xBQj?usp=sharing)""")
          
          
          
          if username=="kuriminellivasu@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP17]")
            st.info("VI12293 Water Observatory - Monitoring water levels of lakes and reservoirs using satellite imagery.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("vasu: kuriminellivasu@gmail.com")
                st.write("Harsha Vardhan: harshavasu6699@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model that can identify surface water levels of waterbodies across the globe Refer this source https://www.blue-dot-observatory.com/aboutwaterobservatory. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset  https://water.blue-dot-observatory.com/api/waterbodies You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Bhuvi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1Lpm4G4xA2IqB2AhtC6BOBmaOMSLk2ged?usp=sharing)""") 
                
          if username=="harshavasu6699@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP17]")
            st.info("VI12293 Water Observatory - Monitoring water levels of lakes and reservoirs using satellite imagery.")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Group Members"):
                st.write("vasu: kuriminellivasu@gmail.com")
                st.write("Harsha Vardhan: harshavasu6699@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""Create a model that can identify surface water levels of waterbodies across the globe Refer this source https://www.blue-dot-observatory.com/aboutwaterobservatory. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset  https://water.blue-dot-observatory.com/api/waterbodies You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Bhuvi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("8/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 20,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1Lpm4G4xA2IqB2AhtC6BOBmaOMSLk2ged?usp=sharing)""")      
            
          if username=="vechalapugovardhan2002@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP18]")  
            
          if username=="prasadmv6305@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP19]")
            st.info("VI12006 Calorie Estimation - Estimate calories from photos of food")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("mr_venkat: prasadmv6305@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""Create a deep learning model that can identify food image nad estimate calories. You can use any deeplearning approach. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a dataset  https://github.com/jubins/DeepLearning-Food-Image-Recognition-And-Calorie-Estimation/tree/master/DataSets You can use this data. You can also add some more data.")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Pranvi")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1-i71rMNv4N8iGwNubMmCRoYphwDqNDle?usp=sharing)""")  
                
          if username=="sahithinalla2304@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP22]") 
            st.info("VI12134 Seedlings - Plant Seedlings Classification from kaggle competition")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Sahithi Nalla : sahithinalla2304@gmail.com")
                st.write("Vavilapalli Vineetha : vineetha1022@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to use any classification models to classify seedlings. Build a best classification model. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/mfsatya/PlantSeedlings-Classification/tree/master/Dataset  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
                
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("3/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 15,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1K1aujkdV4HeVgOkbm2SCVlFKGdPX47O7?usp=sharing)""")      
          
          if username=="vineetha1022@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP22]") 
            st.info("VI12134 Seedlings - Plant Seedlings Classification from kaggle competition")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Sahithi Nalla : sahithinalla2304@gmail.com")
                st.write("Vavilapalli Vineetha : vineetha1022@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to use any classification models to classify seedlings. Build a best classification model. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/mfsatya/PlantSeedlings-Classification/tree/master/Dataset  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
                
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("3/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 15,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1K1aujkdV4HeVgOkbm2SCVlFKGdPX47O7?usp=sharing)""")      
          
          if username=="prahaladh8074995565@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP24]")
            st.info("VI12033 Orders - Order cancellation prediction for hotels")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Prahaladh2001 : prahaladh8074995565@gmail.com")
                st.write("prasanth  gupta : 201801330022@cutmap.ac.in")
                st.write("G.Ganesh : 201801330037@cutmap.ac.in")
                st.write("Pilla Manindhar : manindharpilla@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to predict order cancellations of a restaurant or hotel. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/Hasan330/Order-Cancellation-Prediction-Model/blob/master/distorted_data.csv  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("TARGET2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1IH3iyWUhNsHgMbHRdmvfTOqQhzMgDWoi?usp=sharing)""")      
          
          if username=="201801330022@cutmap.ac.in":
            st.write("Dear user your GROUP INVITE is [GVIOP24]")
            st.info("VI12033 Orders - Order cancellation prediction for hotels")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Prahaladh2001 : prahaladh8074995565@gmail.com")
                st.write("prasanth  gupta : 201801330022@cutmap.ac.in")
                st.write("G.Ganesh : 201801330037@cutmap.ac.in")
                st.write("Pilla Manindhar : manindharpilla@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to predict order cancellations of a restaurant or hotel. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/Hasan330/Order-Cancellation-Prediction-Model/blob/master/distorted_data.csv  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1IH3iyWUhNsHgMbHRdmvfTOqQhzMgDWoi?usp=sharing)""")      
          
          if username=="201801330037@cutmap.ac.in":
            st.write("Dear user your GROUP INVITE is [GVIOP24]")
            st.info("VI12033 Orders - Order cancellation prediction for hotels")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Prahaladh2001 : prahaladh8074995565@gmail.com")
                st.write("prasanth  gupta : 201801330022@cutmap.ac.in")
                st.write("G.Ganesh : 201801330037@cutmap.ac.in")
                st.write("Pilla Manindhar : manindharpilla@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to predict order cancellations of a restaurant or hotel. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/Hasan330/Order-Cancellation-Prediction-Model/blob/master/distorted_data.csv  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1IH3iyWUhNsHgMbHRdmvfTOqQhzMgDWoi?usp=sharing)""")      
          
          
          if username=="manindharpilla@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP24]")
            st.info("VI12033 Orders - Order cancellation prediction for hotels")
            st.success("You have successfully completed Target-1")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Prahaladh2001 : prahaladh8074995565@gmail.com")
                st.write("prasanth  gupta : 201801330022@cutmap.ac.in")
                st.write("G.Ganesh : 201801330037@cutmap.ac.in")
                st.write("Pilla Manindhar : manindharpilla@gmail.com")
                
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to predict order cancellations of a restaurant or hotel. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/Hasan330/Order-Cancellation-Prediction-Model/blob/master/distorted_data.csv  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Mohan Vikas")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("✔️1-15 days")
            st.caption("16-30 days")
            if st.button("Target2",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1IH3iyWUhNsHgMbHRdmvfTOqQhzMgDWoi?usp=sharing)""")      
          
          
          
          if username=="vemulapallisaipraneeth@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP23]")
            st.info("VI12168 Forest Fire - Forest fire detection through UAV imagery using CNNs")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("V.Sai Praneeth : vemulapallisaipraneeth@gmail.com")
                st.write("R.Manivaran : manivaranronanki@gmail.com")
                st.write("K. Shiva Swamy : konashiva5@gmail.com")
                st.write("D. Vineel Kumar : doddavineel@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""In this project you need to use CNN for forest fires detection. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/LeadingIndiaAI/Forest-Fire-Detection-through-UAV-imagery-using-CNNs/tree/master/data  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
                
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("3/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 15,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1a4Q9N5WUlqcJyhGL_v3ITNbwNJYoM9VI?usp=sharing)""")      
          
          if username=="manivaranronanki@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP23]")
            st.info("VI12168 Forest Fire - Forest fire detection through UAV imagery using CNNs")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("V.Sai Praneeth : vemulapallisaipraneeth@gmail.com")
                st.write("R.Manivaran : manivaranronanki@gmail.com")
                st.write("K. Shiva Swamy : konashiva5@gmail.com")
                st.write("D. Vineel Kumar : doddavineel@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""In this project you need to use CNN for forest fires detection. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/LeadingIndiaAI/Forest-Fire-Detection-through-UAV-imagery-using-CNNs/tree/master/data  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
                
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("3/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 15,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1a4Q9N5WUlqcJyhGL_v3ITNbwNJYoM9VI?usp=sharing)""")      
          
          if username=="konashiva5@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP23]")
            st.info("VI12168 Forest Fire - Forest fire detection through UAV imagery using CNNs")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("V.Sai Praneeth : vemulapallisaipraneeth@gmail.com")
                st.write("R.Manivaran : manivaranronanki@gmail.com")
                st.write("K. Shiva Swamy : konashiva5@gmail.com")
                st.write("D. Vineel Kumar : doddavineel@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to use CNN for forest fires detection. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/LeadingIndiaAI/Forest-Fire-Detection-through-UAV-imagery-using-CNNs/tree/master/data  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
                
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("3/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 15,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1a4Q9N5WUlqcJyhGL_v3ITNbwNJYoM9VI?usp=sharing)""")      
          
          if username=="doddavineel@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP23]")
            st.info("VI12168 Forest Fire - Forest fire detection through UAV imagery using CNNs")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("V.Sai Praneeth : vemulapallisaipraneeth@gmail.com")
                st.write("R.Manivaran : manivaranronanki@gmail.com")
                st.write("K. Shiva Swamy : konashiva5@gmail.com")
                st.write("D. Vineel Kumar : doddavineel@gmail.com")
                
            st.title("Problem Statement:")
            st.write("""In this project you need to use CNN for forest fires detection. You can use any model you want. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Dataset:")
            st.write(" We are providing you a sample dataset https://github.com/LeadingIndiaAI/Forest-Fire-Detection-through-UAV-imagery-using-CNNs/tree/master/data  . You can use this dataset or any other additional data too.  ")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
                
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("3/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 15,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/1a4Q9N5WUlqcJyhGL_v3ITNbwNJYoM9VI?usp=sharing)""")      
          
          
          if username=="chowdaryhemanth66@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP20]")
            st.info("VI12030 Predict Prices - Predict hotel room rates.")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Surya Trinadh : suryasrivalli877@gmail.com")
                st.write("jayaram : geddajayaram4@gmail.com")
                st.write("HEMANTH : chowdaryhemanth66@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to  estimate the price per night of hotels rooms in London . First you have to gather data regarding the hotels and its prices, and also about features that could help in explaining the price. Use machine learning concepts for predication. You can use any approach. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Reference:")
            st.write(" We are providing you a reference  https://nycdatascience.com/blog/student-works/spatial-data-science-applied-arcpy-scikit-learn-for-predicting-hotel-room-price")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")    
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/15nLUcUFn88ru_N1Fq9UnluT5UH-rwy3M?usp=sharing)""")      
          
          
          
          
          if username=="geddajayaram4@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP20]")
            st.info("VI12030 Predict Prices - Predict hotel room rates.")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Surya Trinadh : suryasrivalli877@gmail.com")
                st.write("jayaram : geddajayaram4@gmail.com")
                st.write("HEMANTH : chowdaryhemanth66@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to  estimate the price per night of hotels rooms in London . First you have to gather data regarding the hotels and its prices, and also about features that could help in explaining the price. Use machine learning concepts for predication. You can use any approach. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Reference:")
            st.write(" We are providing you a reference  https://nycdatascience.com/blog/student-works/spatial-data-science-applied-arcpy-scikit-learn-for-predicting-hotel-room-price")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")      
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/15nLUcUFn88ru_N1Fq9UnluT5UH-rwy3M?usp=sharing)""")      
          
          if username=="suryasrivalli877@gmail.com":
            st.write("Dear user your GROUP INVITE is [GVIOP20]")
            st.info("VI12030 Predict Prices - Predict hotel room rates.")
            st.caption("Group Project")
            
            with st.expander("Members"):
                st.write("Surya Trinadh : suryasrivalli877@gmail.com")
                st.write("jayaram : geddajayaram4@gmail.com")
                st.write("HEMANTH : chowdaryhemanth66@gmail.com")
                
                
            st.title("Problem Statement:")
            st.write("""In this project you need to  estimate the price per night of hotels rooms in London . First you have to gather data regarding the hotels and its prices, and also about features that could help in explaining the price. Use machine learning concepts for predication. You can use any approach. Deploy your model in streamlit or other frameworks. Your app/web application to be user friendly.""")    
                
            st.title("Reference:")
            st.write(" We are providing you a reference  https://nycdatascience.com/blog/student-works/spatial-data-science-applied-arcpy-scikit-learn-for-predicting-hotel-room-price")
            st.title("Sample Documentation:")
            with st.expander("Sample Documentation"):
                st.caption("This is only template for doucmentation submission. You can adjust based on your project you are doing. You are free to add some other data too.")
                mdlit("""📌About the lifecycle of the project.""")
                mdlit("""📌Importance of defining an objective or goal of the project.""")
                mdlit("""📌Collecting data based on the requirements of the project.""")
                mdlit("""📌Model training and results exploration including:""")
                mdlit("""    📌Establishing baselines for better results.""")
                mdlit("""    📌Adopting techniques and approaches from the existing open-source state-of-the-art models research papers and code repositories.""")
                mdlit("""    📌Experiment tracking and management management """)
                mdlit("""📌Model refinement techniques to avoid underfitting and overfitting like:""")
                mdlit("""    📌Controlling hyperparameters""")
                mdlit("""    📌Regularisation""")
                mdlit("""    📌Pruning""")
                mdlit("""📌Use case diagrams and pipelines.""")
                mdlit("""📌Testing and evaluating your project before deployment.""")
                mdlit("""📌Model deployment""")
                mdlit("""📌Project maintenance""")
            st.info("Mentor : Vikranth")      
            st.title("TO_DO_LIST")
            tabh1,tabh2,tabh3,tabh4=st.tabs(["To_Do","Done","In_progress","Backlog"])
            with tabh1:
                tx=st.text_area("Task To Do",key="jc01")
                n=st.text_input("add a name")
                if st.button("Upload",key="td1"):
                    db1.put({"key":username, "to_do"+n:tx})
                    st.success("Uploaded")
            
            with tabh2:    
                txm=st.text_area("Task Done")
                nm=st.text_input("add a name",key="jcl1")
                if st.button("Upload",key="td2"):
                    db1.put({"key":username, "done"+nm:txm})
                    st.success("Uploaded")
                
            with tabh3:    
                txmd=st.text_area("Task in progress")
                nmd=st.text_input("add a name",key="jcp1")
                if st.button("Upload",key="td3"):
                    db1.put({"key":username, "inprog"+nmd:txmd})
                    st.success("Uploaded")
                
            with tabh4:    
                txmdd=st.text_area("Task backlog")
                nmdd=st.text_input("add a name",key="jcu1")
                if st.button("Upload",key="td4"):
                    db1.put({"key":username, "backlg"+nmdd:txmdd})
                    st.success("Uploaded")
            
            
            if st_toggle_switch("Attendance",key="paypi"):
                today = date.today()
                n=datetime,date.today()
                st.write("Attendance captured:", today)
                #db1.put({"key":username,"Day"+str(n):"p"})
                #st.write("7/90")
                option = {
                    "tooltip": {
                        "formatter": '{a} <br/>{b} : {c}%'
                    },
                    "series": [{
                        "name": 'ATTENDANCE',
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "progress": {
                            "show": "true"
                        },
                        "radius":'100%', 

                        "itemStyle": {
                            "color": '#58D9F9',
                            "shadowColor": 'rgba(0,138,255,0.45)',
                            "shadowBlur": 10,
                            "shadowOffsetX": 2,
                            "shadowOffsetY": 2,
                            "radius": '55%',
                        },
                        "progress": {
                            "show": "true",
                            "roundCap": "true",
                            "width": 15
                        },
                        "pointer": {
                            "length": '100%',
                            "width": 8,
                            "offsetCenter": [0, '5%']
                        },
                        "detail": {
                            "valueAnimation": "true",
                            "formatter": '{value}/90',
                            "backgroundColor": '#58D9F9',
                            "borderColor": '#999',
                            "borderWidth": 4,
                            "width": '60%',
                            "lineHeight": 20,
                            "height": 20,
                            "borderRadius": 188,
                            "offsetCenter": [0, '40%'],
                            "valueAnimation": "true",
                        },
                        "data": [{
                            "value": 19,
                            "name": 'ATTENDANCE'
                        }]
                    }]
                };
                st_echarts(options=option, key="1")
            st.title("Progress:")
            st.caption("1-15 days")
            if st.button("Track1",key="hj"):
                mdlit("""[Upload](https://drive.google.com/drive/folders/15nLUcUFn88ru_N1Fq9UnluT5UH-rwy3M?usp=sharing)""")      
          
       #   if st_toggle_switch("Attendance",key="paypi"):
       #     today = date.today()
       #     st.write("Attendance captured:", today)
          authenticator.logout("Logout","main")
    
   
   
   
   
    
    
    
