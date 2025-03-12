import streamlit as st
from summarizer import MultiDocSummarizer
from evaluation import evaluate_summary

st.title("Multi-Document Summarization & Evaluation")

# Upload text files
uploaded_texts = st.file_uploader("Upload Text Files", accept_multiple_files=True, type="txt")

if st.button("Summarize"):
    if uploaded_texts:
        # Read text files
        original_text = ""
        for file in uploaded_texts:
            original_text += file.read().decode("utf-8") + "\n"

        # Generate Summary
        summarizer = MultiDocSummarizer()
        generated_summary = summarizer.summarize(original_text)

        # Display Summary
        st.subheader("📜 Generated Summary")
        st.write(generated_summary)

        # Evaluate Summary
        metrics = evaluate_summary(original_text, generated_summary)

        # Display Evaluation Metrics
        st.subheader("Evaluation Metrics")
        for metric, (value, range_info) in metrics.items():
            st.write(f"**{metric}:** {value:.4f}  {range_info}")

    else:
        st.warning("Please upload at least one text file.")