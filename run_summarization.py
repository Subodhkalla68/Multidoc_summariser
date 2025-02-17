from data_loader import load_and_merge_documents
from summarizer import MultiDocSummarizer

if __name__ == "__main__":
    # Load and merge documents
    data_dir = "data"
    merged_text = load_and_merge_documents(data_dir)

    # Summarize the merged text
    summarizer = MultiDocSummarizer(model_name="t5-base", max_length=512)
    summary = summarizer.summarize(merged_text, min_length=100, max_length=300)

    # Print the result
    print("=== Multi-Document Summary ===")
    print(summary)