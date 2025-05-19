import os
import re
import numpy as np
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Extractors ---------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported file type. Please use PDF or DOCX.")

# --------- TextRank Summarizer ---------
def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

def build_similarity_matrix(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix

def textrank_summary(sentences, num_sentences=8, damping=0.85, max_iter=100, tol=1e-4):
    sim_matrix = build_similarity_matrix(sentences)
    n = len(sentences)
    scores = np.ones(n)

    for _ in range(max_iter):
        prev_scores = scores.copy()
        for i in range(n):
            scores[i] = (1 - damping) + damping * np.sum((sim_matrix[:, i] * scores) / np.sum(sim_matrix, axis=1))
        if np.linalg.norm(scores - prev_scores) < tol:
            break

    ranked_indices = np.argsort(scores)[-num_sentences:]
    ranked_indices.sort()
    return [sentences[i] for i in ranked_indices]

# --------- Main ---------
if __name__ == "__main__":
    path = input("Enter path to PDF or DOCX file: ").strip('"').strip("'")
    try:
        text = load_text_from_file(path)
        print(f"\nExtracted {len(text)} characters.\n")
        sentences = split_into_sentences(text)
        if len(sentences) < 8:
            print("Text too short to summarize. Showing full content:")
            print("\n".join(sentences))
        else:
            print("Generating summary...\n")
            summary = textrank_summary(sentences, num_sentences=8)
            print("\n--- SUMMARY (8 lines) ---\n")
            for line in summary:
                print(line)
    except Exception as e:
        print(f"Error: {e}")
