📌 AI Echo: Sentiment Analysis of ChatGPT Reviews
📖 Project Overview

AI Echo is a Natural Language Processing (NLP) project that analyzes user reviews of a ChatGPT application and classifies them into Positive, Neutral, or Negative sentiments.
The project also provides business insights through Exploratory Data Analysis (EDA) and an interactive Streamlit dashboard.

🎯 Objectives

Classify user reviews into sentiment categories

Understand customer satisfaction and feedback trends

Identify common issues and improvement areas

Visualize insights for business decision-making

🗂️ Dataset Description

The dataset contains user-generated reviews with the following features:

review – User feedback text

rating – Rating from 1 (very poor) to 5 (excellent)

platform – Web or Mobile

location – User country

version – ChatGPT version

verified_purchase – Verified or non-verified user

review_length – Length of the review text

📌 Note: The dataset is preprocessed before analysis.

🔧 Data Preprocessing

Removed special characters, punctuation, and stopwords

Converted text to lowercase

Applied tokenization and lemmatization

Handled missing values

Converted ratings into sentiment labels

📊 Exploratory Data Analysis (EDA)

The following insights were extracted:

Rating distribution analysis

Sentiment distribution (Positive / Neutral / Negative)

Platform-wise and location-wise rating comparison

Verified vs non-verified user satisfaction

Review length vs rating patterns

Version-wise user satisfaction trends

Word cloud for negative reviews

🤖 Model Building

Text Vectorization: TF-IDF

Algorithm Used: Logistic Regression

Sentiment Classes: Positive, Neutral, Negative

📈 Model Evaluation

Accuracy

Precision

Recall

F1-score

Confusion Matrix

📌 The model achieved strong performance on the dataset.

🌐 Streamlit Dashboard

An interactive Streamlit dashboard was developed to:

Visualize EDA results

Explore sentiment trends

Support business decision-making

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NLTK, Scikit-learn, Matplotlib, WordCloud

Framework: Streamlit

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit App
streamlit run app.py

📌 Project Structure
AI_Echo_Sentiment_Analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── model_training.py
│
└── data/
    └── cleaned_reviews.csv

✅ Key Outcomes

Accurate sentiment classification of user reviews

Actionable insights for customer experience improvement

End-to-end NLP pipeline from preprocessing to deployment

🚀 Future Enhancements

Deploy the app using Streamlit Cloud or AWS

Integrate deep learning models (LSTM / BERT)

Perform real-time sentiment analysis on new reviews

👤 Author

Aravinth Ram

📄 License

This project is for academic and learning purposes.
