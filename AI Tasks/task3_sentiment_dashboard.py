import streamlit as st
from textblob import TextBlob

def main():
    st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="🧠", layout="centered")
    
    st.title("🧠 Sentiment Analysis Dashboard")
    st.write("Analyze the sentiment of social media posts, product reviews, or any text using Natural Language Processing.")
    
    # User input
    st.subheader("Input Text")
    user_input = st.text_area("Enter your text here:", height=150, placeholder="E.g., I absolutely love this new phone! The camera is amazing.")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Process using TextBlob
            blob = TextBlob(user_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment category
            if polarity > 0.1:
                sentiment = "Positive 😊"
                color = "green"
            elif polarity < -0.1:
                sentiment = "Negative 😞"
                color = "red"
            else:
                sentiment = "Neutral 😐"
                color = "gray"
                
            # Display Results
            st.subheader("Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Sentiment**<br><span style='color:{color}; font-size:24px;'>{sentiment}</span>", unsafe_allow_html=True)
            with col2:
                st.metric("Polarity Score", f"{polarity:.2f}", help="-1.0 to 1.0")
            with col3:
                st.metric("Subjectivity Score", f"{subjectivity:.2f}", help="0.0 (Objective) to 1.0 (Subjective)")
                
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
