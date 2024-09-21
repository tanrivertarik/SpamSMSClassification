
# 📱 SMS Spam Classification with Logistic Regression

This repository contains an SMS spam classification project where we build a machine learning model to classify SMS messages as either **spam** or **ham** (not spam). The project demonstrates the application of **Natural Language Processing (NLP)** techniques to a common text classification problem using the **Logistic Regression** model.

## 📚 Project Overview

In this project, we use a dataset of SMS messages that are labeled as either "spam" or "ham." The goal is to develop a machine learning model that can accurately classify new, unseen messages into one of these two categories. 

Key components of the project include:
- **Data Preprocessing**: Cleaning the text data (removing special characters, converting to lowercase, etc.).
- **Text Vectorization**: Converting the raw SMS text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- **Modeling**: Training a **Logistic Regression** classifier and comparing it with other algorithms like **Naive Bayes**.
- **Model Evaluation**: Assessing model performance with metrics such as accuracy, precision, recall, and F1-score.

## 🛠 Features

- **Data Preprocessing**: Includes steps like text cleaning and encoding labels.
- **Vectorization**: Text data is transformed into numeric format using TF-IDF.
- **Model Training**: Logistic Regression is used to train the model, and performance is evaluated.
- **Visualization**: Data distribution and feature importance visualized using bar charts and word clouds.
- **Test Function**: A custom function that allows you to test individual SMS messages.

## 🧠 Machine Learning Models

- **Logistic Regression**: A linear model used for binary classification.
- **Multinomial Naive Bayes** (optional): Another popular algorithm for text classification.
- **Hyperparameter Tuning** (optional): You can try different hyperparameters using GridSearchCV to improve the model.

## 🚀 Technologies Used

- **Python**
- **pandas**, **numpy**: Data manipulation and analysis.
- **scikit-learn**: Machine learning model building and evaluation.
- **TF-IDF Vectorizer**: For converting text to numerical features.
- **matplotlib**, **seaborn**: Data visualization.
- **WordCloud**: For generating word clouds from text data.

## 📊 Visualizations

- **Class Distribution**: A bar chart showing the distribution of "ham" vs. "spam" messages.
- **Word Clouds**: Word clouds for the most frequent words in spam and ham messages.
- **Model Comparison**: Bar chart comparing accuracy of Logistic Regression and Naive Bayes.


## ⚙️ How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tanrivertarik/SpamSMSClassification.git
   cd SMS-Spam-Classifier
   ```

2. **Install dependencies**:
   Make sure you have `pip` installed. Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   If the dataset is not included, you can download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

4. **Run the Jupyter Notebook**:
   Open the `SMS_Spam_Classification.ipynb` notebook in Jupyter or any Python IDE to follow along with the code and execute the steps.
   ```bash
   jupyter notebook SMS_Spam_Classification.ipynb
   ```

5. **Test your own messages**:
   Use the test function provided to predict whether a message is spam or ham. For example:
   ```python
   test_sms = "Congratulations! You've won a free trip to Bahamas. Call now!"
   result = test_message(test_sms, logreg_model, tfidf_vectorizer)
   print(f"The message '{test_sms}' is classified as: {result}")
   ```

## 📈 Results

- The **Logistic Regression** model achieved an accuracy of **97%** on the test set.
- **Precision** and **recall** for spam messages are both high, indicating that the model does a good job of identifying spam.
- The **Naive Bayes** model was also tested and performed similarly well on this dataset.

## 🔥 Future Enhancements

Some ideas for future improvement:
- **Feature Engineering**: Adding more text-based features like message length, number of punctuation marks, and number of digits.
- **Hyperparameter Tuning**: Use GridSearchCV to optimize the Logistic Regression model.
- **Advanced Models**: Explore deep learning-based models such as LSTM or transformers (e.g., BERT) for text classification.
- **Model Deployment**: Create a web interface using Flask or Streamlit to deploy the spam classification model for real-time predictions.


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
