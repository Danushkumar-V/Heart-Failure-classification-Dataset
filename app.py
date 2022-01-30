import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import typeconv as tp
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode


model = pickle.load(open('predic_model.pkl','rb'))

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_xnlvopiy.json")
lottie_hospital = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_vPnn3K.json")
# st_lottie(lottie_hello,speed=1,reverse=False,loop=True,quality="high", renderer="svg", height=None,width=None,key=None,)

@st.cache( allow_output_mutation=True )
def get_data(filepath):
    data = pd.read_csv(filepath)
    return data



error_rate = [0.3442028985507246, 0.39492753623188404, 0.30434782608695654, 0.30434782608695654, 0.27898550724637683, 0.2971014492753623, 0.3007246376811594, 0.31521739130434784, 0.2898550724637681, 0.31521739130434784, 0.2826086956521739, 0.3115942028985507, 0.3007246376811594, 0.30434782608695654, 0.29347826086956524, 0.30434782608695654, 0.3007246376811594, 0.3079710144927536, 0.30434782608695654, 0.30434782608695654, 0.3115942028985507, 0.3188405797101449, 0.3079710144927536, 0.3079710144927536, 0.3115942028985507, 0.322463768115942, 0.32971014492753625, 0.31521739130434784, 0.3115942028985507, 0.3115942028985507, 0.31521739130434784, 0.3007246376811594, 0.3007246376811594, 0.30434782608695654, 0.286231884057971, 0.2971014492753623, 0.29347826086956524, 0.2971014492753623, 0.2826086956521739]


def new_dataframe(new_data):
    b=pd.DataFrame(new_data, columns= ['Age' ,'Sex' ,'ChestPainType','RestingBP' ,'Cholesterol' ,'FastingBS'  ,'RestingECG' ,'MaxHR' ,'ExerciseAngina' ,'Oldpeak','ST_Slope'])
    return b



def EDA(heart):
    fig = make_subplots(rows=5, cols=2, shared_xaxes=False, shared_yaxes=False,
                    subplot_titles=("Distribution of Age<br>by Heart Disease", "", 
                                    "Distribution of Systolic Blood Pressure<br>by Heart Disease",  "",
                                    "Distribution of Cholesterol<br>by Heart Disease",  "", 
                                    "Distribution of Maximum Heart Rate<br>by Heart Disease", "" ,
                                    "Distribution of ST Segment Depression<br>by Heart Disease", ""))

    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==1]['Age'], histnorm='probability density', 
                            marker=dict(color='#AF4343', line=dict(width=1, color='#000000')),
                            opacity=0.8, name="Heart Disease"),
                row=1, col=1)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==0]['Age'], histnorm='probability density', 
                            marker=dict(color='#C6AA97', line=dict(width=1, color='#000000')), 
                            opacity=0.75, name="No Disease"),
                row=1, col=1)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==1]['Age'], name="Heart Disease", 
                        marker_color = '#AF4343', showlegend=False),
                row=1, col=2)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==0]['Age'], name="No Disease", 
                        marker_color = '#C6AA97', showlegend=False), 
                row=1, col=2)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==1]['RestingBP'], histnorm='probability density', 
                            marker=dict(color='#AF4343', line=dict(width=1, color='#000000')),
                            opacity=0.8, name="Heart Disease", showlegend=False),
                row=2, col=1)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==0]['RestingBP'], histnorm='probability density',
                            marker=dict(color='#C6AA97', line=dict(width=1, color='#333333')), 
                            opacity=0.75, name="No Disease", showlegend=False),
                row=2, col=1)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==1]['RestingBP'], name="Heart Disease", 
                        marker_color = '#AF4343', showlegend=False),
                row=2, col=2)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==0]['RestingBP'], name="No Disease", 
                        marker_color = '#C6AA97', showlegend=False), 
                row=2, col=2)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==1]['Cholesterol'], histnorm='probability density', 
                            marker=dict(color='#AF4343', line=dict(width=1, color='#000000')),
                            opacity=0.8, name="Heart Disease", showlegend=False),
                row=3, col=1)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==0]['Cholesterol'], histnorm='probability density', 
                            marker=dict(color='#C6AA97', line=dict(width=1, color='#333333')), 
                            opacity=0.75, name="No Disease", showlegend=False),
                row=3, col=1)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==1]['Cholesterol'], name="Heart Disease", 
                        marker_color = '#AF4343', showlegend=False),
                row=3, col=2)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==0]['Cholesterol'], name="No Disease", 
                        marker_color = '#C6AA97', showlegend=False), 
                row=3, col=2)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==1]['MaxHR'], histnorm='probability density', 
                            marker=dict(color='#AF4343', line=dict(width=1, color='#000000')),
                            opacity=0.8, name="Heart Disease", showlegend=False),
                row=4, col=1)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==0]['MaxHR'], histnorm='probability density', 
                            marker=dict(color='#C6AA97', line=dict(width=1, color='#333333')), 
                            opacity=0.75, name="No Disease", showlegend=False),
                row=4, col=1)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==1]['MaxHR'], name="Heart Disease", 
                        marker_color = '#AF4343', showlegend=False),
                row=4, col=2)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==0]['MaxHR'], name="No Disease", 
                        marker_color = '#C6AA97', showlegend=False), 
                row=4, col=2)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==1]['Oldpeak'], histnorm='probability density', 
                            marker=dict(color='#AF4343', line=dict(width=1, color='#000000')),
                            opacity=0.8, name="Heart Disease", showlegend=False),
                row=5, col=1)
    fig.add_trace(go.Histogram(x=heart[heart.HeartDisease==0]['Oldpeak'], histnorm='probability density', 
                            marker=dict(color='#C6AA97', line=dict(width=1, color='#333333')), 
                            opacity=0.75, name="No Disease", showlegend=False),
                row=5, col=1)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==1]['Oldpeak'], name="Heart Disease", 
                        marker_color = '#AF4343', showlegend=False),
                row=5, col=2)
    fig.add_trace(go.Box(y=heart[heart.HeartDisease==0]['Oldpeak'], name="No Disease", 
                        marker_color = '#C6AA97', showlegend=False), 
                row=5, col=2)
    fig.update_layout(title="Heart Disease Distributions", 
                    xaxis1_title="Age, in years", yaxis1_title='Probability Density', 
                    yaxis2_title='Age, in years',
                    xaxis3_title="Blood Pressure, mmHg", yaxis3_title='Probability Density', 
                    yaxis4_title="Blood Pressure, mmHg",
                    xaxis5_title="Cholesterol, mg/dl", yaxis5_title='Probability Density',
                    yaxis6_title="Cholesterol, mg/dl",
                    xaxis7_title="Heart Rate, bpm", yaxis7_title='Probability Density', 
                    yaxis8_title="Heart Rate, bpm",
                    xaxis9_title="ST Segment Depression, mm", yaxis9_title='Probability Density',
                    yaxis10_title="ST Segment Depression, mm",
                    barmode='overlay', showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=.95),
                    height=2000, width=800)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', zeroline=False)
    fig.update_yaxes(showline=False, zeroline=False)
    return(fig)



header_content = st.container()
dataset_descrb = st.container()
prediction = st.container()

with header_content:
    st.title('Hello Friends :innocent:')
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=True,
        loop=True,
        quality="low", # medium ; high# canvas
        height=400,
        width=500,
        key=None,
    )
    
    st.header("This is an end-to-end model on Heart failure prediction!!")
    first_para = '<p style="font-family:Courier; color:Black; font-size: 17px;">In this project I have worked on a heart failure dataset and train model using machine learning technology that can predict for heart failure cases :)</p>'
    st.markdown(first_para, unsafe_allow_html=True)
    sec_para = '<p style="font-family:Courier; color:Black; font-size: 17px;">Let us make a look on the dataset and its meta data <3 </p>'
    st.markdown(sec_para, unsafe_allow_html=True)

with dataset_descrb:
    st.header('Heart Failure Prediction Dataset!')
    second_para = """<p style="font-family:Courier; color:Black; font-size: 17px;">Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year,
    which accounts for 31% of all deaths worldwide.
    Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age.
    Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.</p>"""
    st.markdown(second_para, unsafe_allow_html=True)
    second_para_1 = """<p style="font-family:Courier; color:Black; font-size: 17px;">People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease)
     need early detection and management wherein a machine learning model can be of great help.<3 </p>"""
    st.markdown(second_para_1, unsafe_allow_html=True)
    # st.area_chart(data=df.head(21), width=1000, height=300, use_container_width=True)
    # with st.expander("See explanation"):
    #  st.write("""
    #      The above chart is derived using the below given data set...
    #  """)
    st_lottie(
        lottie_hospital,
        speed=1,
        reverse=True,
        loop=True,
        quality="low", # medium ; high# canvas
        height=400,
        width=500,
        key=None,
    )
    st.subheader("Source")
    third_para_1 = """<p style="font-family:Courier; color:Black; font-size: 17px;">This dataset was created by combining different datasets already available independently but not combined before. 
    In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. 
    The five datasets used for its curation are:</p>"""
    st.markdown(third_para_1, unsafe_allow_html=True)
    st.write("""
        - *Cleveland: 303 observations*
        - *Hungarian: 294 observations*
        - *Switzerland: 123 observations*
        - *Long Beach VA: 200 observations*
        - *Stalog (Heart) Data Set: 270 observations*
    """)
    fourth_para_1 = """<p style="font-family:Courier; color:Black; font-size: 17px;">
        Total: 1190 observations</p>"""
    fourth_para_2 = """<p style="font-family:Courier; color:Black; font-size: 17px;">
        Duplicated: 272 observations</p>"""
    fourth_para_3 = """<p style="font-family:Courier; color:Black; font-size: 17px;">
        Final dataset: 918 observations</p>"""
    st.markdown(fourth_para_1, unsafe_allow_html=True)
    st.markdown(fourth_para_2, unsafe_allow_html=True)
    st.markdown(fourth_para_3, unsafe_allow_html=True)
    st.subheader("Let's visualize the data!")
    path = "D:\DK\Dev\Heart-Failure-Prediction-Dataset\heart.csv"
    df = get_data(path)
    df2 = tp.typeconvo(df)
    st.write('---')
    fig = EDA(df)
    st.write(fig)
    
    error_para1 = """<p style="font-family:Courier; color:Black; font-size: 17px;">
        Let's see the Error vs KNN value visualization</p>"""
    st.markdown(error_para1, unsafe_allow_html=True)
    fig3 = plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='red', linestyle='dashed', marker='o',
            markerfacecolor='green', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    st.pyplot(fig3)
    error_para2 = """<p style="font-family:Courier; color:Black; font-size: 17px;">
        From this plot we can get to know that k value of 5 has lowest error!...let's train the model with k value of 5:)</p>"""
    st.markdown(error_para2, unsafe_allow_html=True)








with prediction:
    st.header(""" Let's predict ! """)
    third_para = """<p style="font-family:Courier; color:Black; font-size: 17px;">Enter the below values and lets check 
    whether the person has heart failure are not...</p>"""
    st.markdown(third_para, unsafe_allow_html=True)
    a, b = st.columns(2)
    Age = a.text_input('Enter the age of the person:',0)
    Sex = b.selectbox('Enter the gender of the person:', options=['Male','Female'], index = 0)
    ChestPainType = b.selectbox('Enter the Chest pain type:',options = ['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'], index = 0)
    RestingBP = a.text_input('Enter the resting blood pressure [mm Hg]:', 0)
    RestingBP = int(RestingBP)
    Cholesterol = a.text_input('Enter the serum cholesterol [mm/dl]:', 0)
    Cholesterol = int(Cholesterol)
    FastingBS = b.text_input('Enter the fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]:',0)
    FastingBS = int(FastingBS)
    RestingECG = st.selectbox('Enter the resting electrocardiogram results:',options = ["Normal","Showing probable or definite left ventricular hypertrophy by Estes' criteria","Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)"], index=0)

    MaxHR = a.text_input('Enter the maximum heart rate achieved [Numeric value between 60 and 202]:',0)
    MaxHR = int(MaxHR)
    ExerciseAngina = b.selectbox('Enter that exercise-induced angina [Yes/No]:',options=['Yes','No'],index=0)
    Oldpeak = a.slider('Enter the ST [Numeric value measured in depression]:', min_value=-5.0,max_value=7.0,value=1.0,step=0.1)
    ST_Slope = b.selectbox('Enter the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]:',options=['Up','Flat','Down'],index=0)
    new_data = [[Age ,Sex ,ChestPainType ,RestingBP ,Cholesterol ,FastingBS  ,RestingECG ,MaxHR ,ExerciseAngina ,Oldpeak ,ST_Slope]]
    new_df = new_dataframe(new_data)
    new_data_after_typeconv = tp.typeconvo(new_df)

    predict_value = model.predict(new_data_after_typeconv)
    result = st.button("Predict")
    if result:
        if predict_value == 1:
            st.subheader('You have been identified with some heart disease :pensive:')
        else:
            st.subheader('I am very happy to say that your heart is at good condition! :smile:')