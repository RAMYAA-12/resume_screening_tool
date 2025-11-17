# Resume_screening_tool
Python tool to rank resumes based on similarity to a job description using NLP and TF-IDF
# Description

A Python-based tool to automatically screen and rank resumes based on their similarity to a job description. Useful for HR professionals, recruiters, or anyone who wants to filter candidates efficiently.

## Features
- Parses PDF resumes and extracts text.
- Preprocesses resumes and job descriptions (tokenization, stopword removal, lowercase).
- Calculates similarity score using TF-IDF vectorization and cosine similarity.
- Outputs a ranked list of resumes with percentage match.

## Technologies Used
- Python 3.x
- PyMuPDF (`fitz`) for PDF parsing
- NLTK for text preprocessing
- Scikit-learn for TF-IDF and similarity calculation


