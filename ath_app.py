import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x==1, 1, 0)
    return(x)


ss = pd.DataFrame({
   "income":np.where(s["income"]<=9, s["income"], np.nan),
    "education":np.where(s["educ2"]<=8, s["educ2"], np.nan),
    "parent": clean_sm(s["par"]),
    "married": clean_sm(s["marital"]),
    "female":np.where(s["gender"]==2, 1, 0),
    "age":np.where(s["age"]<=98, s["age"], np.nan),
    "sm_li":clean_sm(s["web1h"])
})
ss = ss.dropna()

#st.dataframe(ss)



# #### 4. Create a target vector (y) and feature set (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

 #### 5. Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=312) # set for reproducibility


# ---

# #### 6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[79]:


lr = LogisticRegression()


# In[80]:


# Fit algorithm to training data
lr.fit(X_train, y_train)


# In[ ]:




# #### 10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?


##############################################################################

# STREAMLIT

#############################################################################

st.markdown(" *MSBA Programming 2 - Final Project by: Amina Zimic*")
st.title('Are you a LinkedIn User?')
#st.subheader("by: Amina Zimic")

from PIL import Image
pic = Image.open("pic.webp")

col1, col2 = st.columns(2)
with col1:
    st.text("")
    st.markdown(" ##### Enter your income, education, parental and marital status, gender, and age using the sidebar on the left - and click on the *LinkedIn User?* button below to see our data based prediction!")
    

with col2:
    #st.subheader("by: Amina Zimic")
    st.image(pic, width=300)
    



#st.markdown('text')

#st.header("header")
# col1, col2 = st.columns(2)

# with col1:
#     st.text("Demographic 1")
#     income = st.slider('Income ($)', 1, 9, 1)
#     education = st.slider('Education', 1, 8, 1)
#     parent = st.slider('Parent', 0, 1, 1)

# with col2:
#     st.text("Demographics 2")
#     married = st.slider('Married', 0, 1, 1)
#     gender = st.slider('Female', 0, 1, 1)
#     age = st.slider('Age', 10, 98, 1)

# st.text('')
# if st.button("Predict if a Linkedin User"):
#     result = lr.predict(
#         np.array([[income, education, parent, married, gender, age]]))
#     st.write("Predicted class: ", result[0])
#     prob = lr.predict_proba(
#         np.array([[income, education, parent, married, gender, age]]))
#     st.write("Probability that this person is a Linkedin User: ", prob[0][1])
    


with st.sidebar: 
# Income
    "#### What is your yearly income?"
    income = st.selectbox("Please select income:", 
             options = ["Less than $10,000",
                        "$10,000 to $20,000",
                        "$20,000 to $30,000",
                        "$30,000 to $40,000",
                        "$40,000 to $50,000",
                        "$50,000 to $75,000",
                        "$75,000 to $100,000",
                        "$100,000 to $150,000",
                        "$150,000 or more"
                        ])


if income == "Less than $10,000":
    income = 1
elif income == "$10,000 to $20,000":
    income = 2
elif income == "$20,000 to $30,000":
    income = 3
elif income == "$30,000 to $40,000":
    income = 4
elif income == "$40,000 to $50,000":
    income = 5
elif income == "$50,000 to $75,000":
    income = 6
elif income == "$75,000 to $100,000":
    income = 7
elif income == "$100,000 to $150,000":
    income = 8
else:
    income = 9
    
with st.sidebar:
# Education
    "#### What is the highest level of school/degree you completed?"
    education = st.selectbox("Please select education level:", 
             options = ["Less than high school (Grades 1-8 or no formal schooling)",
                        "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                        "High school graduate (Grade 12 with diploma or GED certificate)",
                        "Some college, no degree (includes some community college)",
                        "Two-year associate degree from a college or university",
                        "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                        "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
                        ])


if education == "Less than high school (Grades 1-8 or no formal schooling)":
    education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education = 3
elif education == "Some college, no degree (includes some community college)":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
else:
    education = 8

with st.sidebar:
# Parent
    "#### Are you a parent of a child under 18 living in your home?"
    parent = st.selectbox("Please select Yes or No:", 
             options = ["Yes",
                        "No",                        
                        ])


if parent == "Yes":
    parent = 1
else:
    parent = 0

with st.sidebar:
# Married
    "#### What is your current marital status?"
    married = st.selectbox("Please select marital status:", 
             options = ["Married",
                        "Living with a partner", 
                        "Divorced",
                        "Separated",
                        "Widowed",
                        "Never been married",                       
                        ])


if married == "Married":
    married = 1
elif married == "Living with a partner":
    married = 0
elif married == "Divorced":
    married = 0
elif married == "Separated":
    married = 0
elif married == "Widowed":
    married = 0
else:
    married = 0


with st.sidebar:
# Gender
    "#### What is your gender?"
    gender = st.selectbox("Please select gender:", 
             options = ["Female",
                        "Male",                        
                        ])


if gender == "Female":
    gender = 1
else:
    gender = 0

with st.sidebar:
# Age
    "#### How old are you?"
    age = st.number_input("Please enter your age:", min_value=1, max_value=98)


st.text('')
if st.button("LinkedIn User?"):
    result = lr.predict(
        np.array([[income, education, parent, married, gender, age]]))
    
    # if result == 1:
    #     result = "Yes"
    # else:
    #     result = "No"
    
    #st.write("Predicted LinkedIn user? ", result[0])
    prob = lr.predict_proba(
        np.array([[income, education, parent, married, gender, age]]))
    #st.write("Probability that you are a LinkedIn User: ", round(prob[0][1], 2))


#st.write(f"This person is {mar_label}, a {deg_label}, and in a {inc_label} bracket")

#### Create label (called sent) from TextBlob polarity score to use in summary below



    df = pd.DataFrame({
        "income": [income],
        "education": [education],
        "parent": [parent],
        "married": [married],
        "female": [gender],
        "age": [age]
    })


    probability = lr.predict_proba(df)

    if probability[0][1] > 0.5:
        label = "You are a LinkedIn User"
    else:
        label = "You are not a LinkedIn User"    

    #### Show results

    ### Print sentiment score, label, and language
    st.markdown(f"## Predicted class: **{label}**")

    #probability = 0.6434
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability[0][1],
        title = {'text': f"Probability that you are a LinkedIn user is: {round((probability[0][1]*100))}%"},
        gauge = {"axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0, 1], "color":"tomato"},
                    {"range": [0.5, 1], "color":"darkblue"}
                ],
                "bar":{"color":"silver"}}
    ))


    st.plotly_chart(fig)




st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.caption("Image source: https://www.proinfluent.com/en/who-uses-linkedin/")