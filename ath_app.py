import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


st.markdown("# Who is a Linkedin User? - by Amina Zimic")

st.markdown("### Please enter the information below")

name = "Amina Zimic"

st.write(name)

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

alt_plot = alt.Chart(ss.groupby(["age", "sm_li"], as_index=False)["sm_li"].mean()).mark_circle().encode(x="age",
      y="sm_li",
      color="sm_li:N")
#st.altair_chart(alt_plot, use_container_width=True)


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



# New data for features: income, education, parent, married, female and age
#person_1 = [8, 7, 0, 1, 1, 42]

# Predict class, given input features
#predicted_class_1 = lr.predict([person_1])

# Generate probability of positive class (=1)
#probs_1 = lr.predict_proba([person_1])


# In[120]:


# Print predicted class and probability
#print(f"Predicted class: {predicted_class_1[0]}") # 0=not Linkedin User, 1=Linkedin User
#print(f"Probability that this person is a Linkedin User: {probs_1[0][1]}")


# In[121]:


# New data for features: income, education, parent, married, female and age
#person_2 = [8, 7, 0, 1, 1, 82]

# Predict class, given input features
#predicted_class_2 = lr.predict([person_2])

# Generate probability of positive class (=1)
#probs_2 = lr.predict_proba([person_2])


# In[122]:


# Print predicted class and probability
#print(f"Predicted class: {predicted_class_2[0]}") # 0=not Linkedin User, 1=Linkedin User
#print(f"Probability that this person is a Linkedin User: {probs_2[0][1]}")


###################################

"### What is your yearly income?"
# answerInc = st.selectbox(label="Please select",
# options=(" ",
#     "Less than $10,000", "$10,000 to $20,000", "$20,000 to $30,000", 
#         "$30,000 to $40,000", "$40,000 to $50,000", "$50,000 to $75,000", 
#         "$75,000 to $100,000", "$100,000 to $150,000",
#     "$150,000 or more"))
# st.write("You selected: ", answerInc)


# inputs = []
# for each in options:
#     inputs.append((each, 
#         if answerInc == "Less than $10,000":
#             print(1)
#         elif answerInc == "$10,000 to $20,000":
#             print(2)
#         elif answerInc == "$20,000 to $30,000":
#             print(3)
#         elif answerInc == "$30,000 to $40,000":
#             print(4)
#         elif answerInc == "$40,000 to $50,000":
#             print(5)
#         elif answerInc == "$50,000 to $75,000":
#             print(6)
#         elif answerInc == "$75,000 to $100,000":
#             print(7)
#         elif answerInc == "$100,000 to $150,000":
#             print(8)
#         elif answerInc == "$150,000 or more":
#             print(9)
#         elif answerInc == " ":
#             print(98)
#         else:
#             print(99)
#     )



###### 5 Take user input

# "### User input"
# st.text_input("What do you like most about analytics?", key="analytics_like")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from prediction import predict


"### What is your yearly income?"
answerInc = st.selectbox(label="Please select",
options=(" ",
    "Less than $10,000", "$10,000 to $20,000", "$20,000 to $30,000", 
        "$30,000 to $40,000", "$40,000 to $50,000", "$50,000 to $75,000", 
        "$75,000 to $100,000", "$100,000 to $150,000",
    "$150,000 or more"))
st.write("You selected: ", answerInc)

st.title('Classifying Linkedin Users')
st.markdown('intro')

st.header("User Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Demographic 1")
    income = st.slider('Income ($)', 1, 9, 1)
    education = st.slider('Education', 1, 8, 1)
    parent = st.slider('Parent', 0, 1, 1)

with col2:
    st.text("Demographics 2")
    married = st.slider('Married', 0, 1, 1)
    gender = st.slider('Female', 0, 1, 1)
    age = st.slider('Age', 10, 98, 1)

st.text('')
if st.button("Predict if a Linkedin User"):
    result = lr.predict(
        np.array([[income, education, parent, married, gender, age]]))
    st.write("Predicted class: ", result[0])
    prob = lr.predict_proba(
        np.array([[income, education, parent, married, gender, age]]))
    st.write("Probability that this person is a Linkedin User: ", prob[0][1])
    


 
    
#probs = lr.predict_proba([result])

# # Predict class, given input features
# predicted_class_2 = lr.predict([person_2])

# # Generate probability of positive class (=1)
# probs_2 = lr.predict_proba([person_2])