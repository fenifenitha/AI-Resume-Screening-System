# 🚀 AI Resume Screening & Candidate Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

# 📌 Project Overview

The **AI Resume Screening & Candidate Recommendation System** is an intelligent recruitment automation platform that helps HR teams and recruiters evaluate resumes efficiently using **Artificial Intelligence, Natural Language Processing (NLP), and Machine Learning**.

The system automatically analyzes resumes, extracts technical skills, compares candidates against job requirements, calculates compatibility scores, ranks applicants, and generates hiring recommendations.

This project eliminates the need for manual resume screening and enables faster, data-driven hiring decisions.

---

# 🎯 Problem Statement

Recruiters often receive hundreds of resumes for a single job opening.

Manual screening is:

- Time-consuming
- Error-prone
- Inconsistent
- Difficult to scale

This project addresses these challenges by automating the resume evaluation process using AI-powered techniques.

---

# 💡 Solution

The system performs:

✅ Resume Parsing

✅ Resume Classification

✅ Skill Extraction

✅ Job Description Matching

✅ Candidate Ranking

✅ Recommendation Generation

✅ Resume Analysis Dashboard

---

# ✨ Key Features

## 📄 Resume Parsing

- Extracts text from PDF resumes
- Supports multiple resume formats
- Converts unstructured resume content into analyzable text

---

## 🧹 NLP-Based Text Preprocessing

The system cleans resume text using:

- Lowercase conversion
- URL removal
- Number removal
- Punctuation removal
- Stopword removal
- Tokenization

This improves machine learning performance.

---

## 🧠 Skill Extraction Engine

Automatically identifies candidate skills including:

### Programming Languages

- Python
- Java
- C
- C++
- JavaScript
- TypeScript
- PHP
- Ruby
- Swift

### Web Development

- HTML
- CSS
- React
- Angular
- Node.js
- Flask
- Django

### Databases

- SQL
- MySQL
- PostgreSQL
- MongoDB

### Data Science & AI

- Machine Learning
- Deep Learning
- NLP
- TensorFlow
- PyTorch
- OpenCV
- Scikit-Learn
- Data Science

### Cloud & DevOps

- AWS
- Azure
- Docker
- Kubernetes

### Analytics Tools

- Power BI
- Tableau
- Excel

### Development Tools

- Git
- GitHub
- Linux

---

## 🤖 Resume Classification

The system predicts resume categories using Machine Learning.

### Models Used

### Logistic Regression

- Fast and efficient
- Excellent baseline model

### Random Forest Classifier

- Ensemble learning technique
- High prediction accuracy
- Robust against overfitting

---

## 📊 Resume Matching System

The system compares resumes against job descriptions using:

### TF-IDF Vectorization

Converts textual information into numerical feature vectors.

### Cosine Similarity

Measures similarity between:

- Resume
- Job Description

Output:

```text
Match Percentage = Similarity Score × 100
```

---

## 🏆 Candidate Ranking

The system ranks all candidates according to:

- Resume relevance
- Skill match
- Similarity score

Top candidates are automatically identified.

---

## 💼 Candidate Recommendation Engine

The recommendation engine evaluates:

### Matched Skills

Skills present in both:

- Resume
- Job Requirements

### Missing Skills

Required skills absent in candidate profile.

### Recommendation Categories

| Score | Recommendation |
|---------|---------------|
| 80%+ | Highly Recommended |
| 60% - 79% | Moderately Recommended |
| Below 60% | Needs Skill Improvement |

---

## 🌐 Interactive Streamlit Web Application

The project includes a user-friendly web interface built with Streamlit.

Users can:

- Upload PDF resumes
- Enter job descriptions
- Analyze candidate profiles
- View match percentage
- View extracted skills
- Get hiring recommendations instantly

---

# 🏗️ System Architecture

```text
                ┌─────────────────┐
                │ Resume PDF      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Extraction │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ NLP Processing  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Skill Extraction│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ TF-IDF Features │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Cosine Similarity│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Match Score     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Recommendation  │
                └─────────────────┘
```

---

# 🛠️ Technology Stack

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- Logistic Regression
- Random Forest

## Natural Language Processing

- NLTK
- TF-IDF Vectorizer

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Web Framework

- Streamlit

## PDF Processing

- PyPDF2

## Dataset Source

- Kaggle Resume Dataset

---

# 📂 Project Structure

```text
AI-Resume-Screening-System/
│
├── app.py
├── resume.py
├── requirements.txt
├── README.md
│
├── data/
│   └── UpdatedResumeDataSet.csv
│
├── screenshots/
│   ├── home.png
│   ├── results.png
│   └── ranking.png
│
└── assets/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📈 Machine Learning Workflow

```text
Resume Dataset
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train-Test Split
      │
      ▼
Machine Learning Models
      │
      ├── Logistic Regression
      └── Random Forest
      │
      ▼
Prediction & Evaluation
```

---

# 📊 Sample Output

```text
Match Percentage: 87.42%

Extracted Skills:
['python', 'sql', 'machine learning', 'tensorflow']

Matched Skills:
['python', 'sql', 'machine learning']

Missing Skills:
['nlp']

Recommendation:
Highly Recommended
```

---

# 🔍 Evaluation Metrics

The system evaluates model performance using:

- Accuracy Score
- Classification Report
- Precision
- Recall
- F1 Score

---

# 🚀 Future Enhancements

- BERT-Based Resume Analysis
- Sentence Transformers
- Resume Summarization
- ATS Compatibility Scoring
- Multi-Resume Bulk Screening
- Recruiter Dashboard
- AI Interview Question Generator
- Generative AI Candidate Feedback
- Resume Improvement Suggestions
- LLM-Based Hiring Assistant

---

# 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing
- Machine Learning
- Resume Analytics
- Candidate Ranking Systems
- Streamlit Application Development
- Text Similarity Analysis
- Data Preprocessing
- Feature Engineering
- Recruitment Automation

---

# 🤝 Contribution

Contributions, suggestions, and feature requests are welcome.

Feel free to fork this repository and create a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👩‍💻 Author

### Fenitha T K

**B.Tech Computer Science Engineering**

Aspiring Data Scientist | AI Enthusiast | Machine Learning Practitioner | NLP Learner

📧 Email: your-email@example.com

🔗 LinkedIn: your-linkedin-profile

🐙 GitHub: your-github-profile

---

## ⭐ If you found this project useful, please give it a star on GitHub!
