import streamlit as st
import pickle
import numpy as np

# ============================
# Load Trained Assets
# ============================
model = pickle.load(open("final_best_model.pkl", "rb"))
vectorizer = pickle.load(open("final_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ========================================
# Streamlit UI
# ========================================
st.set_page_config(page_title="Department Prediction App", page_icon="🛍️")

st.title("🛍️ Clothing Department Prediction")
st.write("Enter a customer review, and the system will predict the product department.")

# Text input
user_text = st.text_area("Enter Review Text:", height=180)

if st.button("Predict Department"):
    if user_text.strip() == "":
        st.warning("Please enter a review text!")
    else:
        # TF-IDF transform
        X = vectorizer.transform([user_text])

        # Prediction
        pred = model.predict(X)[0]

        # Decode label
        department = label_encoder.inverse_transform([pred])[0]

        # Prediction probabilities (if supported)
        try:
            proba = model.predict_proba(X)[0]
            prob_df = {
                label_encoder.inverse_transform([i])[0]: round(float(p)*100, 2)
                for i, p in enumerate(proba)
            }
        except:
            prob_df = None

        # Display results
        st.success(f"### Predicted Department: **{department}**")

        if prob_df is not None:
            st.subheader("Confidence Scores")
            st.json(prob_df)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
