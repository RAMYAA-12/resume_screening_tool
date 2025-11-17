import os
import fitz  # PyMuPDF
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job description
with open("job_description.txt", "r") as file:
    job_desc = file.read()

# Preprocess text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(words)

job_desc_clean = preprocess(job_desc)

# Load resumes from folder
resume_folder = "resumes/"
results = []

for file_name in os.listdir(resume_folder):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(resume_folder, file_name)
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        cleaned_resume = preprocess(text)

        # Vectorize and calculate similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([job_desc_clean, cleaned_resume])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        results.append((file_name, round(score * 100, 2)))  # As percentage

# Display sorted results
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
print("\n📋 Resume Screening Results:\n")
for name, score in sorted_results:
    print(f"{name}: {score}% match")