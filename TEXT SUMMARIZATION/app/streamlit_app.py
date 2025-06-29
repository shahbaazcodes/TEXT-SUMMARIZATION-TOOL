import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

st.set_page_config(page_title="📝 Text Summarizer", layout="centered")

st.markdown("<h1 style='text-align: center;'>📝 Text Summarization Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Summarize any text and download the result</p>", unsafe_allow_html=True)

text_input = st.text_area("✏️ Enter your text to summarize:", height=300)
MAX_INPUT_LEN = 1024
summary = ""

if st.button("🚀 Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    elif len(text_input) > MAX_INPUT_LEN:
        st.error(f"⚠️ Text too long! Please keep input under {MAX_INPUT_LEN} characters.")
    else:
        try:
            with st.spinner("Loading model..."):
                device = torch.device("cpu")
                model_name = "sshleifer/distilbart-cnn-12-6"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
            with st.spinner("Generating summary..."):
                result = summarizer(text_input, max_length=100, min_length=30, do_sample=False)
                summary = result[0]['summary_text']
                st.success("✅ Summary generated!")
        except Exception as e:
            st.error(f"❌ Error during summarization: {e}")

if summary:
    st.markdown("### 📌 Summary:")
    st.write(summary)
    st.download_button("💾 Download Summary as .txt", data=summary, file_name="summary.txt", mime="text/plain")
