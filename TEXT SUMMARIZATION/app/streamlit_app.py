import streamlit as st
from transformers import pipeline

# Set page config
st.set_page_config(page_title="📝 Text Summarizer", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>📝 Text Summarization Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Summarize any text and download the result</p>", unsafe_allow_html=True)

# Text input
text_input = st.text_area("✏️ Enter your text to summarize:", height=300)

# Summarizer model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summary = ""
if st.button("🚀 Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            result = summarizer(text_input, max_length=100, min_length=30, do_sample=False)
            summary = result[0]['summary_text']
            st.success("✅ Summary generated!")

# Show summary
if summary:
    st.markdown("### 📌 Summary:")
    st.write(summary)

    # Download button
    st.download_button(
        label="💾 Download Summary as .txt",
        data=summary,
        file_name="summary.txt",
        mime="text/plain",
    )


