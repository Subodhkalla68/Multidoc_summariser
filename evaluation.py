import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForMaskedLM
from rouge_score import rouge_scorer
import textstat  # For readability scores

# Load the sentence transformer for computing text similarity
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load BERT tokenizer and model for computing perplexity
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def compute_perplexity(text):
    """
    Calculate the perplexity of the generated summary using BERT.
    Lower perplexity = better fluency.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return np.exp(loss)  # Perplexity = exp(loss)

def compute_rouge_scores(original_text, generated_summary):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.
    Higher ROUGE = better content retention.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(original_text, generated_summary)
    
    rouge_metrics = {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }
    return rouge_metrics

def evaluate_summary(original_text, generated_summary):
    """
    Evaluates the generated summary using:
    - Context Relevance Score (cosine similarity)
    - Perplexity Score (language fluency)
    - Compression Ratio (length reduction)
    - ROUGE Scores (text similarity)
    - Readability Score (Flesch-Kincaid)
    """

    # Compute embeddings for original text and summary
    original_embedding = embedder.encode(original_text, convert_to_tensor=True)
    summary_embedding = embedder.encode(generated_summary, convert_to_tensor=True)

    # Ensure embeddings have correct shape
    if original_embedding.dim() == 1:
        original_embedding = original_embedding.unsqueeze(0)
    if summary_embedding.dim() == 1:
        summary_embedding = summary_embedding.unsqueeze(0)

    # Compute Context Relevance Score (Cosine Similarity)
    similarity_score = util.pytorch_cos_sim(original_embedding, summary_embedding).item()

    # Compute Perplexity Score
    perplexity = compute_perplexity(generated_summary)

    # Compute Compression Ratio (how much text is reduced)
    compression_ratio = len(generated_summary) / len(original_text)

    # Compute ROUGE Scores
    rouge_scores = compute_rouge_scores(original_text, generated_summary)

    # Compute Readability Score (Flesch-Kincaid Score)
    readability_score = textstat.flesch_reading_ease(generated_summary)

    # Define range explanations
    ranges = {
        "Context Relevance Score": "(Range: 0-1, higher is better)",
        "Perplexity Score": "(Range: ~10-100, lower is better)",
        "Compression Ratio": "(Ideal: ~0.2-0.5, lower means more compression)",
        "ROUGE-1 Score": "(Range: 0-1, higher is better)",
        "ROUGE-2 Score": "(Range: 0-1, higher is better)",
        "ROUGE-L Score": "(Range: 0-1, higher is better)",
        "Readability Score": "(Range: 0-100, higher = easier to read, >60 is good)"
    }

    # Compile evaluation results
    metrics = {
        "Context Relevance Score": (round(similarity_score, 4), ranges["Context Relevance Score"]),
        "Perplexity Score": (round(perplexity, 4), ranges["Perplexity Score"]),
        "Compression Ratio": (round(compression_ratio, 4), ranges["Compression Ratio"]),
        "ROUGE-1 Score": (round(rouge_scores["ROUGE-1"], 4), ranges["ROUGE-1 Score"]),
        "ROUGE-2 Score": (round(rouge_scores["ROUGE-2"], 4), ranges["ROUGE-2 Score"]),
        "ROUGE-L Score": (round(rouge_scores["ROUGE-L"], 4), ranges["ROUGE-L Score"]),
        "Readability Score": (round(readability_score, 2), ranges["Readability Score"])
    }

    return metrics