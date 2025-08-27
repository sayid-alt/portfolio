
---

# Sentiment Analysis NLP App

This is a Streamlit application for sentiment analysis of user feedback, criticism, and suggestions. The app automatically detects negative, neutral, or positive sentiments based on input data.

## Setup Instructions

Follow these steps to set up and run the app locally:

### 1. Clone the Repository
Navigate to the directory where you want to clone the repository, then clone it.

```bash
git clone https://github.com/sayid-alt/sentiment-analysis-nlp.git
```

### 2. Navigate to the Project Directory
```bash
cd sentiment-analysis-nlp
```

### 3. Create a Python Virtual Environment
Make sure you have Python 3.10 installed, then create a virtual environment.

```bash
python3.10 -m venv venv
```

### 4. Activate the Virtual Environment
- **For Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **For macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 5. Install the Required Libraries
Ensure `pip` is up to date and install the necessary dependencies from `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Streamlit App
Once the dependencies are installed, you can run the Streamlit app.

```bash
streamlit run app.py
```

### 7. Access the Application
Once the app is running, you can access it via your web browser at `http://localhost:8501`.

---

