import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
data=pd.read_csv("diabetes.csv")
print(data.head())
print(data.columns)

print(data.isnull().sum)

correlation_matrix=data.corr()
print(correlation_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("matrice di correlazione")
plt.show()
#seleziono le colonne più utili
features=["Glucose","BMI","Age","Pregnancies"]
X=data[features]
y=data["Outcome"]
#divido i dati in train e test
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
#creo il modello
model=LogisticRegression()
model.fit(X_train,y_train)

#faccio le previsioni
y_pred=model.predict(X_test)

#valuto il modello
print("Accuracy",accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
import streamlit as st


# UI Streamlit
st.title("Diabetes Risk Predictor")
st.write("Inserisci i dati del paziente:")

glucose = st.slider("Glicemia", 0, 200, 100)
bmi = st.slider("BMI", 10.0, 70.0, 25.0)
age = st.slider("Età", 10, 100, 30)
pregnancies = st.slider("Numero di gravidanze", 0, 20, 1)

# Predizione
input_data = pd.DataFrame([[glucose, bmi, age, pregnancies]], columns=features)
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0][1]
st.write(f"Probabilità di diabete:{proba:.2%}")
if st.button("Valuta rischio"):
    if prediction == 1:
        st.error("Alta probabilità di diabete!")
    else:
        st.success("Bassa probabilità di diabete.")