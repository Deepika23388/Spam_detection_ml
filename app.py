import streamlit as st
import pickle

# 1. Page Configuration (Browser tab title and icon)
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# 2. Load the trained Model and TF-IDF Vectorizer
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# 3. Web Page UI Headers
st.title("📩 SMS Spam Message Detector")
st.write("Enter any SMS or text message below to check whether it is **SPAM** or **NOT SPAM (HAM)**.")

# 4. Sidebar with Project Details (Great for Interviews/Demos)
with st.sidebar:
    st.header("ℹ️ Project Details")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Feature Extraction:** TF-IDF Vectorizer")
    st.write("**Accuracy:** ~97.85%")
    st.write("**Tech Stack:** Python, Scikit-learn, Streamlit")
    st.markdown("---")
    st.info("💡 Tip: Try testing messages like 'Win a free lottery' or 'Call me when you are free'.")

# 5. User Input Area
user_input = st.text_area("Enter your message here:", placeholder="Type or paste your SMS message here...", height=120)

# 6. Prediction Button
if st.button("🔍 Check Message", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first before checking!")
    else:
        # Step A: Transform input text to numbers using saved TF-IDF
        input_data = tfidf.transform([user_input])
        
        # Step B: Predict with Model
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Probabilities: [ham_prob, spam_prob]
        spam_confidence = probabilities[1] * 100
        ham_confidence = probabilities[0] * 100

        # Step C: Display Results
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **SPAM MESSAGE DETECTED!**")
            st.write(f"Confidence: **{spam_confidence:.2f}%** spam probability.")
        else:
            st.success(f"✅ **NOT SPAM (HAM MESSAGE)**")
            st.write(f"Confidence: **{ham_confidence:.2f}%** normal message probability.")