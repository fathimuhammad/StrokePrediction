import joblib
import numpy as np
import pandas as pd
import time as timer
import seaborn as sns
import streamlit as st

from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_score,recall_score,f1_score


#model imports
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

#charts
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.markdown(
        f"""
        <style>
            .block-container{{
                min-width: 65vw;
            }}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            footer:after {{
                visibility: visible;
                display: flex;
                justify-content: center;
                color: white;
                background-color: #f63366;
                padding: -5px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.title("STROKE PREDICTION AND ANALYSIS USING MACHINE LEARNING")

    menu = ["Prediction","Model Details"]
    activity = st.selectbox("Select a Menu",menu)

    if activity == "Model Details":

        st.subheader("Cleaned Dataset")
        #loading dataframe
        df = pd.read_csv("train.csv")

        #Filling Missing values of bmi using mean
        df['bmi'] = df['bmi'].fillna(df.groupby('age')['bmi'].transform('mean'))
        df.fillna(df[['bmi']].mean(),inplace=True)
        
        #Dropping ID attribute
        df.drop(['id'],axis=1,inplace=True)
        st.dataframe(df)
        
        st.subheader("Data Visualizations")

        #Heart disease vs stroke histogram
        fig = px.histogram(df, x = df['heart_disease'], y = df['stroke'],labels={
            'heart_disease':"Heart Disease",
            'stroke':"Number of Strokes",
        }, title='Heart Disease Vs Stroke')
        st.write(fig)

        #histogram buttons
        col1, col2, col3 = st.beta_columns(3)
        col4, col5, col6 = st.beta_columns(3)

        #Age vs stroke histogram
        if col1.button("Age Vs Occurance of Stroke"):
            fig = px.histogram(df, x = df['age'], y = df['stroke'],labels={
                "age":"Age",
            }, title='Age Vs Occurance of Stroke')
            st.write(fig)

        #Smoking Status vs stroke histogram
        if col2.button("Smoking Status Vs Occurance of Stroke"):
            fig = px.histogram(df, x = df['smoking_status'], y = df['stroke'],labels={
                "smoking_status":"Smoking Status",
            }, title='Smoking Status Vs Occurance of Stroke')
            st.write(fig)

        #Hypertension vs stroke histogram
        if col3.button("Hypertension Vs Occurance of Stroke"):
            fig = px.histogram(df, x = df['hypertension'], y = df['stroke'],labels={
                "hypertension":"Hypertension",
            }, title='Hypertension Vs Occurance of Stroke')
            st.write(fig)

        #BMI vs stroke histogram      
        if col4.button("BMI vs Occurance of Stoke"):
            fig = px.histogram(df, x = df['bmi'], y = df['stroke'],labels={
                "bmi":"BMI",
            }, title='BMI Vs Occurance of Stroke')
            st.write(fig)

        #Average Glucose level vs stroke histogram
        if col5.button("Average Glucose Level vs Occurance of Stoke"):
            fig = px.histogram(df, x = df['avg_glucose_level'], y = df['stroke'],labels={
                "avg_glucose_level":"Average Glucose Level",
            }, title='Average Glucose Level Vs Occurance of Stroke')
            st.write(fig)

        #other factiors vs stroke histogram as subplots
        if col6.button("Other Factors vs Occurance of Stoke"):
            fig = make_subplots(rows=2, cols=2)
            
            #Marital Status vs stroke histogram
            fig.add_trace(
                go.Histogram(x=df['ever_married'], y=df['stroke'],name='Marital Status vs Stroke'),
                row=1, col=1
            )
        
            #Residence Type vs stroke histogram
            fig.add_trace(
                go.Histogram(x=df['Residence_type'],y=df['stroke'],name='Residence Type vs Stroke'),
                row=1, col=2
            )

            #Work Type vs stroke histogram
            fig.add_trace(
                go.Histogram(x=df['work_type'],y=df['stroke'],name='Work Type vs Stroke'),
                row=2, col=1
            )

            #Gender vs stroke histogram
            fig.add_trace(
                go.Histogram(x=df['gender'],y=df['stroke'],name='Gender vs Stroke'),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Marital Status", row=1, col=1)
            fig.update_yaxes(title_text="Stroke", row=1, col=1)
            fig.update_xaxes(title_text="Residence Type", row=1, col=2)
            # fig.update_yaxes(title_text="Stroke", row=1, col=2)
            fig.update_xaxes(title_text="Work Type", row=2, col=1)
            fig.update_yaxes(title_text="Stroke", row=2, col=1)
            fig.update_xaxes(title_text="Gender", row=2, col=2)
            # fig.update_yaxes(title_text="Stroke", row=2, col=2)
            fig.update_layout(height=600, width=800, title_text="Factors vs Stroke")
            st.write(fig)

        #Encoding categorical attributes to values
        label_gender = LabelEncoder()
        label_married = LabelEncoder()
        label_work = LabelEncoder()
        label_residence = LabelEncoder()
        label_smoking = LabelEncoder()
        df['gender'] = label_gender.fit_transform(df['gender'])
        df['ever_married'] = label_married.fit_transform(df['ever_married'])
        df['work_type']= label_work.fit_transform(df['work_type'])
        df['Residence_type']= label_residence.fit_transform(df['Residence_type'])
        df['smoking_status']= label_smoking.fit_transform(df['smoking_status'])
        
        st.subheader("Data Frame after Encoding categorical attributes to values")
        st.dataframe(df)

        #Class Distribution Pie Chart
        class_occur = df['stroke'].value_counts()
        class_names = ['No Stroke','Stroke']
        fig, ax = plt.subplots(figsize=(8,4))
        ax.pie(class_occur, labels=class_names, autopct='%1.2f%%',
                shadow=True, startangle=0, counterclock=False)
        ax.axis('equal')
        ax.set_title('Class Distribution')
        st.pyplot(fig,figsize=(1, 1))
        st.write("No Stroke: {}".format(class_occur[0]))
        st.write("Stroke: {}".format(class_occur[1]))

        #Handling Imbalanced Class Data Using SMOTE Technique
        smote = SMOTE(sampling_strategy='minority')
        X, y= smote.fit_resample(df.loc[:,df.columns!='stroke'], df['stroke'])
        
        #Class Distribution Pie Chart after using SMOTE Technique
        _, class_counts = np.unique(y, return_counts=True)
        class_names = ['No stroke', 'Stroke']
        fig, ax = plt.subplots(figsize=(8,4))
        ax.pie(class_counts, labels=class_names, autopct='%1.2f%%',
                shadow=True, startangle=90, counterclock=False)
        ax.axis('equal') 
        ax.set_title('Class Distribution after performing SMOTE')
        st.write(fig)
        st.write("No Stroke: {}".format(class_counts[0]))
        st.write("Stroke: {}".format(class_counts[1]))

        #Correlation Matrix
        st.subheader("Correlation Matrix Color Map")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.matshow(df.corr())
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_yticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns,rotation=90)
        ax.set_yticklabels(df.columns)

        #Creating colorbar for Correlation Matrix
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=10)
        fig.tight_layout()
        st.write(fig)

        #Correlation Matrix HeatMap
        st.subheader("Correlation Matrix Heat Map")
        f, ax = plt.subplots(figsize = (6,4))
        corr = df.corr()
        hm = sns.heatmap(round(corr,2), annot=True, annot_kws={"size":8}, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
        f.subplots_adjust(top=0.93)
        t=f.suptitle("Correlation Heatmap",fontsize=10)
        st.write(f)

        #Building Data Model and Training
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=42)

        #Data Standardization
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train_std = scaler.transform(X_train)
        X_test_std = scaler.transform(X_test)

        #ML Model Training and Evaluation
        model_menu = ["XGBoost (XGB) with HyperTuned Parameters","XGBoost (XGB)","Random Forest (RF)","Logistic Regression (LR)","Decision Tree (DT)","Gaussian Naive Bayes (GNB)","Singular Vector Machine (SVM)"]
        model = st.selectbox("Select a Model",model_menu)
        #XGBoost
        if model == "XGBoost (XGB)":
            start = timer.time()
            xgb_m = XGBClassifier(objective="reg:logistic", random_state=42,use_label_encoder = False)
            xgb_m.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_xgb = xgb_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_xgb)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_xgb))
            st.write("Precision:",metrics.precision_score(y_test, y_xgb))
            st.write("Recall:",metrics.recall_score(y_test, y_xgb))
            st.write("F1:",metrics.f1_score(y_test, y_xgb))
        #XGBoost with HyperTuned Parameter
        elif model == "XGBoost (XGB) with HyperTuned Parameters":
            start = timer.time()
            xgb_mt = XGBClassifier(objective="reg:logistic", random_state=42,
                                use_label_encoder = False, colsample_bytree= 0.5, 
                                gamma= 0.2, learning_rate= 0.25,
                                max_depth= 10, min_child_weight= 1,)
            xgb_mt.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_xgb = xgb_mt.predict(X_test)
            y_train_predict = xgb_mt.predict(X_train)
            cnf_matrix = metrics.confusion_matrix(y_train , y_train_predict)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write('Train Accuracy',accuracy_score(y_train , y_train_predict))
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_xgb))
            st.write("Precision:",metrics.precision_score(y_test, y_xgb))
            st.write("Recall:",metrics.recall_score(y_test, y_xgb))
            st.write("F1:",metrics.f1_score(y_test, y_xgb))
        # Random Forest
        elif model == "Random Forest (RF)":
            start = timer.time()
            ranfor_m = RandomForestClassifier(n_estimators=100, random_state=42)
            ranfor_m.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_ranfor = ranfor_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_ranfor)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_ranfor))
            st.write("Precision:",metrics.precision_score(y_test, y_ranfor))
            st.write("Recall:",metrics.recall_score(y_test, y_ranfor))
            st.write("F1:",metrics.f1_score(y_test, y_ranfor))
        # Decision Tree
        elif model == "Decision Tree (DT)":
            start = timer.time()
            dtree_m = DecisionTreeClassifier(random_state=42)
            dtree_m.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_dtree = dtree_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_dtree)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_dtree))
            st.write("Precision:",metrics.precision_score(y_test, y_dtree))
            st.write("Recall:",metrics.recall_score(y_test, y_dtree))
            st.write("F1:",metrics.f1_score(y_test, y_dtree))
        # Logistic Regression
        elif model == "Logistic Regression (LR)":
            start = timer.time()
            logit_m = LogisticRegression(solver='lbfgs', random_state=42)
            logit_m.fit(X_train_std,y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_pred = logit_m.predict(X_test_std)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            st.write("Precision:",metrics.precision_score(y_test, y_pred))
            st.write("Recall:",metrics.recall_score(y_test, y_pred))
            st.write("F1:",metrics.f1_score(y_test, y_pred))
        # Gaussian Naive Bayes
        elif model == "Gaussian Naive Bayes (GNB)":
            start = timer.time()
            gnb_m = GaussianNB()
            gnb_m.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_gnb = gnb_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_gnb)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_gnb))
            st.write("Precision:",metrics.precision_score(y_test, y_gnb))
            st.write("Recall:",metrics.recall_score(y_test, y_gnb))
            st.write("F1:",metrics.f1_score(y_test, y_gnb))
        # Singular Vector Machine
        elif model == "Singular Vector Machine (SVM)":
            start = timer.time()
            svm_m = SVC(kernel='rbf',probability=True)
            svm_m.fit(X_train_std, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_svm = svm_m.predict(X_test_std)
            cnf_matrix = metrics.confusion_matrix(y_test, y_svm)
            st.write("Confusion matrix:\n",cnf_matrix)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_svm))
            st.write("Precision:",metrics.precision_score(y_test, y_svm))
            st.write("Recall:",metrics.recall_score(y_test, y_svm))
            st.write("F1 Score:",metrics.f1_score(y_test, y_svm))
        
        st.info("XGBoost (with Hyper Tuned Parameters) has been selected as the Best Model due to its High Accuracy compared to other models that has been Trained")

    #Prediction Page
    if activity == 'Prediction': 

        st.markdown("Enter the User's Details to predict the occurance of Stroke")
        st.text("Please Enter correct details to get better results")
        
        #Getting User Inputs
        gender = st.radio("What is User's gender",("Male","Female"))
        age = st.number_input("Enter User's age",value=40)
        hypertension = st.radio("Hypertension?",("Yes","No"))
        heart_disease = st.radio("User Ever had a heart disease?",("Yes","No"))
        ever_married = st.radio("User Ever Married?",("Yes","No"))
        work_type = st.radio("What is User's work type?",("Government Job","Private Job","Self Employed","Never Worked","Children"))
        Residence_type = st.radio("What is User's Residence type?",("Urban","Rural"))
        avg_glucose_level = st.number_input("Enter User's Average Glucose Level",value=92.35)
        
        #BMI Calculation with Height and Weight is User doesn't know BMI
        if st.checkbox("Dont Know BMI? Use height and weight"):
            height = st.number_input("Enter User's Height in cm",value=160)
            weight = st.number_input("Enter User's Weight in kgs",value=60)
            bmi = weight / (height/100)**2
            st.write("BMI of user is {:.2f} and will be autoupdated".format(bmi))
        else:
            bmi = st.number_input("Enter User's BMI",value=25.4)

        smoking_status = st.radio("User's Smoking Status?",("Unknown","Formerly Smoked","Never Smoked","Smokes"))
        
        #model (XGBoost)
        prediction_model = 'XGBoost'
        trained_model = joblib.load('Models/XGBoostTunedModel.pkl')
        model_accuracy = "94.9%"

        if st.button("Submit"):
            #Encoding categorical attributes to values
            gender = 1 if gender == 'Male' else 0
            age = float(age)
            hypertension = 1 if hypertension == 'Yes' else 0
            ever_married = 1 if ever_married == 'Yes' else 0
            heart_disease = 1 if heart_disease == 'Yes' else 0
            if work_type == 'Government Job':
                work_type = 0 
            elif work_type == 'Never Worked':
                work_type = 1 
            elif work_type == 'Private Job':
                work_type = 2 
            elif work_type == 'Self Employed':
                work_type = 3
            elif work_type == 'Children':
                work_type = 4 
            Residence_type = 1 if Residence_type == 'Urban' else 0
            avg_glucose_level = float(avg_glucose_level)
            bmi = float(bmi)
            if smoking_status == 'Unknown':
                smoking_status = 0 
            elif smoking_status == 'Formerly Smoked':
                smoking_status = 1 
            elif smoking_status == 'Never Smoked':
                smoking_status = 2
            elif smoking_status == 'Smokes':
                smoking_status = 3 

            #Creating nparray of User Inputs
            user_input = np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1,-1)
            
            #converting into dataframe to avoid mismatching feature_names error
            user_input = pd.DataFrame(user_input, columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
            
            #prediction using selected model
            prediction = trained_model.predict(user_input)

            #Prediction Probability
            pred_prob = trained_model.predict_proba(user_input)
            stroke_prob = pred_prob[0][1]*100

            #Printing Predicted results
            if prediction == 1:
                st.header("User has Higher Chances of having a Stroke😔")
            else:
                st.header("User has Lower Chances of having a Stroke😊")
            
            #printing prediction probability 
            if stroke_prob < 25:
                st.success("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            elif stroke_prob < 50:
                st.info("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            elif stroke_prob < 75:
                st.warning("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            else:
                st.error("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            st.text("Predicted with "+prediction_model+" Model with Accuracy of " +model_accuracy)
    
if __name__ == '__main__':
    main()
