import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the trained model
try:
    with open("linear_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ✅ UI Layout
st.title("📱 Instagram Likes Predictor")
st.write("Enter the values to predict the number of likes.")

# ✅ User input fields
followers = st.number_input("Follows", min_value=0)
comments = st.number_input("Comments", min_value=0)
shares = st.number_input("Shares", min_value=0)
saves = st.number_input("Saves", min_value=0)
profile_visits = st.number_input("Profile Visits", min_value=0)
from_hashtags = st.number_input("From Hashtags", min_value=0)

# ✅ Data Upload Option
st.subheader("📂 Upload Your Custom Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Uploaded Dataset:**", df.head())

    # ✅ Data Visualization using Seaborn
    st.subheader("📈 Dataset Trends Visualization")
    fig, ax = plt.subplots()
    sns.pairplot(df)
    st.pyplot(fig)

# ✅ Prediction & Scatter Plot
if st.button("Predict Likes"):
    input_data = pd.DataFrame([[followers, comments, shares, saves, profile_visits, from_hashtags]],
                              columns=["Follows", "Comments", "Shares", "Saves", "Profile Visits", "From Hashtags"])
    
    try:
        prediction = model.predict(input_data)
        predicted_likes = int(prediction[0])
        st.success(f"🔥 Predicted Likes: {predicted_likes}")

        # ✅ Scatter Plot for Prediction
        st.subheader("📊 Likes Prediction Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(["Predicted Likes"], [predicted_likes], color="red", label="Predicted")
        ax.set_ylabel("Number of Likes")
        ax.set_title("Predicted Likes Visualization")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
