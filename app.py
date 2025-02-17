import streamlit as st
from summarizer import MultiDocSummarizer
from data_loader import load_and_merge_documents

st.title("Multi-Document Summarizer")
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type="txt")

if st.button("Summarize"):
    # Save uploaded files temporarily
    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.read())
    
    # Merge documents and summarize
    merged_text = load_and_merge_documents("data")
    summarizer = MultiDocSummarizer()
    summary = summarizer.summarize(merged_text)

    st.subheader("Summary")
    st.write(summary)