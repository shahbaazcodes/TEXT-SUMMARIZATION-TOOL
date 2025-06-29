import streamlit as st
from transformers import pipeline

# Set page config
st.set_page_config(page_title="📝 Text Summarizer", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>📝 Text Summarization Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Summarize any text and download the result</p>", unsafe_allow_html=True)

# Text input
text_input = st.text_area("✏️ Enter your text to summarize:", height=300)

# Safe input length limit (to prevent memory error on Streamlit Cloud)
MAX_INPUT_LEN = 1024

summary = ""
if st.button("🚀 Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    elif len(text_input) > MAX_INPUT_LEN:
        st.error(f"⚠️ Text too long! Please keep input under {MAX_INPUT_LEN} characters.")
    else:
        try:
            with st.spinner("Generating summary..."):
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                result = summarizer(text_input, max_length=100, min_length=30, do_sample=False)
                summary = result[0]['summary_text']
                st.success("✅ Summary generated!")
        except Exception as e:
            st.error(f"❌ Error during summarization: {e}")

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
