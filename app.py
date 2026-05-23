import streamlit as st
import re

from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Skill Database
# -----------------------------

skills_list = [
    'python',
    'java',
    'sql',
    'machine learning',
    'deep learning',
    'tensorflow',
    'nlp',
    'react',
    'html',
    'css',
    'javascript'
]

# -----------------------------
# Clean Resume
# -----------------------------

def clean_resume(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    return text

# -----------------------------
# Extract Skills
# -----------------------------

def extract_skills(resume):

    found_skills = []

    for skill in skills_list:

        if skill in resume:
            found_skills.append(skill)

    return found_skills

# -----------------------------
# Extract PDF Text
# -----------------------------

def extract_text_from_pdf(pdf_file):

    text = ""

    pdf_reader = PdfReader(pdf_file)

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("AI Resume Screening System")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if uploaded_file is not None:

        resume_text = extract_text_from_pdf(
            uploaded_file
        )

        cleaned_resume = clean_resume(
            resume_text
        )

        candidate_skills = extract_skills(
            cleaned_resume
        )

        documents = [
            cleaned_resume,
            job_description
        ]

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        match_percentage = similarity * 100

        st.subheader("Results")

        st.write(
            f"Match Percentage: {match_percentage:.2f}%"
        )

        st.write("Extracted Skills:")

        st.write(candidate_skills)

    else:

        st.warning("Please upload a PDF resume")