import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Load your trained model 
# ---------------------------
model = joblib.load("model.pkl")   # change filename if needed
vectorizer = joblib.load("vectorizer.pkl")         # only if you saved vectorizer separately

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Email / SMS Spam Detector", layout="wide")

st.title("📧 Email / SMS Spam Detection App")
st.write("Enter any message below to check whether it is **Spam** or **Ham**.")

# Text input
user_input = st.text_area("Enter your message here:", height=180)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Transform text
        transformed_text = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(transformed_text)[0]

        # Show result
        if prediction == 1 or prediction == "spam":
            st.error("🚨 **This message is classified as SPAM!**")
        else:
            st.success("📨 **This message is HAM (Not Spam).**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
