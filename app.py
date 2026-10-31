
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Venkatesh | Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

st.title("Human Activity Recognition System")
st.subheader("Machine Learning Based Activity Detection")

st.markdown("""
### Developed by Venkatesh S
**Department of Electronics and Communication Engineering**  
**Dhanalakshmi College of Engineering**
""")

st.divider()

st.write("""
This project uses sensor data and Machine Learning
to recognize human activities such as walking,
running, sitting and standing.
""")

st.info("Upload your labelled sensor dataset in CSV format.")

uploaded_file = st.file_uploader(
    "Upload Sensor Dataset",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    required = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "activity"
    ]

    if all(col in data.columns for col in required):
        data = data[required].dropna()

        X = data.drop(columns=["activity"])
        y = data["activity"]

        st.success("Dataset uploaded successfully!")

        st.write("### Dataset Preview")
        st.dataframe(data.head())

        if y.nunique() < 2 or y.value_counts().min() < 2:
            st.error("Dataset needs at least two activities with two samples each.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

            st.write("### Activity Distribution")
            st.bar_chart(data["activity"].value_counts())

            st.write("### Predict Activity")

            values = [
                st.number_input(col, value=0.0)
                for col in X.columns
            ]

            if st.button("Predict Activity"):
                sample = pd.DataFrame([values], columns=X.columns)
                result = model.predict(sample)[0]
                st.success(f"Predicted Activity: {result}")

    else:
        st.error("Please upload a CSV with all seven required columns.")

st.divider()

st.caption("Project by Venkatesh S | ECE | Dhanalakshmi College of Engineering")
