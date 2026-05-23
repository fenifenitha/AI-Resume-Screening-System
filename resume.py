# -*- coding: utf-8 -*-

# =========================
# INSTALL REQUIRED PACKAGES
# =========================
# Run this in terminal before execution:
# pip install pandas numpy scikit-learn nltk matplotlib kagglehub

import re
import shutil
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import kagglehub

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# DOWNLOAD NLTK DATA
# =========================
nltk.download('stopwords')
nltk.download('punkt')

# =========================
# DOWNLOAD DATASET
# =========================
path = kagglehub.dataset_download(
    "jillanisofttech/updated-resume-dataset"
)

print("Path to dataset files:", path)

# Copy dataset locally
shutil.copytree(path, "data2", dirs_exist_ok=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data2/UpdatedResumeDataSet.csv")

print(df.head())


# =========================
# TEXT CLEANING
# =========================
stop_words = set(stopwords.words('english'))


def clean_resume(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    cleaned_text = " ".join(words)

    return cleaned_text


# Apply cleaning
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

print(df[['Resume', 'cleaned_resume']].head())

print("\nSample Cleaned Resume:\n")
print(df['cleaned_resume'][0])


# =========================
# TF-IDF VECTORIZATION
# =========================
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['cleaned_resume'])

print("\nTF-IDF Shape:", X.shape)

# Labels
y = df['Category']

# Encode labels
le = LabelEncoder()

y = le.fit_transform(y)

print("\nEncoded Labels:")
print(y[:10])


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# =========================
# LOGISTIC REGRESSION MODEL
# =========================
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("\nLogistic Regression Accuracy:", accuracy_lr)

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))


# =========================
# RANDOM FOREST MODEL
# =========================
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", accuracy_rf)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


# =========================
# RESUME MATCHING
# =========================
job_description = """
Looking for a Data Science candidate with
Python, Machine Learning, Deep Learning,
SQL, NLP, TensorFlow, and Data Analysis skills.
"""

resume_text = df['cleaned_resume'][0]

documents = [resume_text, job_description]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

similarity_score = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:2]
)

match_percentage = similarity_score[0][0] * 100

print(
    "\nResume Match Percentage:",
    round(match_percentage, 2),
    "%"
)


# =========================
# MATCH ALL CANDIDATES
# =========================
job_description = """
Looking for Python, Machine Learning,
Data Science, NLP, SQL, TensorFlow skills.
"""

scores = []

for i in range(len(df)):

    resume = df['cleaned_resume'][i]

    documents = [resume, job_description]

    tfidf_matrix = vectorizer.fit_transform(documents)

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    scores.append(score)

df['match_score'] = scores

top_candidates = df.sort_values(
    by='match_score',
    ascending=False
)

print("\nTop Candidates:")
print(
    top_candidates[
        ['Category', 'match_score']
    ].head(10)
)


# =========================
# SKILL EXTRACTION
# =========================
skills_list = [

    # Programming Languages
    'python', 'java', 'c', 'c++',
    'javascript', 'typescript',
    'php', 'ruby', 'swift',

    # Web Development
    'html', 'css', 'react',
    'angular', 'nodejs',
    'flask', 'django',

    # Databases
    'sql', 'mysql',
    'mongodb', 'postgresql',

    # Data Science / AI
    'machine learning',
    'deep learning',
    'data science',
    'nlp',
    'computer vision',
    'tensorflow',
    'keras',
    'pytorch',
    'opencv',
    'scikit-learn',

    # Cloud / DevOps
    'aws', 'azure',
    'docker', 'kubernetes',

    # Analytics
    'power bi',
    'tableau',
    'excel',

    # Tools
    'git',
    'github',
    'linux'
]


def extract_skills(resume):

    found_skills = []

    resume = resume.lower()

    for skill in skills_list:

        if skill in resume:
            found_skills.append(skill)

    return found_skills


sample_resume = df['cleaned_resume'][0]

print("\nExtracted Skills:")
print(extract_skills(sample_resume))

df['skills'] = df['cleaned_resume'].apply(extract_skills)

print("\nSkills Data:")
print(df[['Category', 'skills']].head())


# =========================
# SKILL MATCHING
# =========================
required_skills = [
    'python',
    'machine learning',
    'sql',
    'tensorflow'
]


def skill_match(candidate_skills, required_skills):

    matched = []

    for skill in required_skills:

        if skill in candidate_skills:
            matched.append(skill)

    return matched


candidate_skills = df['skills'][0]

matched_skills = skill_match(
    candidate_skills,
    required_skills
)

print("\nMatched Skills:")
print(matched_skills)

match_percent = (
    len(matched_skills)
    /
    len(required_skills)
) * 100

print(
    "\nSkill Match Percentage:",
    match_percent,
    "%"
)


# =========================
# CANDIDATE RECOMMENDATION
# =========================
def recommend_candidate(
    candidate_skills,
    required_skills,
    match_score
):

    matched = []
    missing = []

    for skill in required_skills:

        if skill in candidate_skills:
            matched.append(skill)
        else:
            missing.append(skill)

    if match_score >= 80:
        recommendation = "Highly Recommended"

    elif match_score >= 60:
        recommendation = "Moderately Recommended"

    else:
        recommendation = "Needs Skill Improvement"

    return {
        "Matched Skills": matched,
        "Missing Skills": missing,
        "Recommendation": recommendation
    }


required_skills = [
    'python',
    'machine learning',
    'sql',
    'tensorflow',
    'nlp'
]

candidate_skills = df['skills'][0]

print("\nCandidate Skills:")
print(candidate_skills)

matched_skills = []

for skill in required_skills:

    if skill in candidate_skills:
        matched_skills.append(skill)

match_score = (
    len(matched_skills)
    /
    len(required_skills)
) * 100

print("\nMatch Score:", match_score)

# FIXED ERROR HERE
result = recommend_candidate(
    candidate_skills,
    required_skills,
    match_score
)

print("\nMatched Skills:")
print(result['Matched Skills'])

print("\nMissing Skills:")
print(result['Missing Skills'])

print("\nFinal Recommendation:")
print(result['Recommendation'])