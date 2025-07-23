# IMDb Sentiment Analysis API

A FastAPI service that serves a fine-tuned DistilBERT model for sentiment classification of movie reviews.

## Project Approach and Methodology

### 1. Requirement
- Data analysis on the IMDB Movie Reviews dataset from Kaggle.
- Tokenization and classification model training.
- REST API return sentiment and probability.
- Docker-only deployment.

### 2. System Design
- Designed modular folder layout with `src/` for logic and `models/` for assets.
- Built data analysis notebooks for exploratory insight, hypothesis testing, and model development.
- Compared multiple classical and deep learning models (Logistic Regression, Naive Bayes, LSTM, BERT).
- Used SHAP for explanation.
- Saved models as `.pkl` or `transformers` formats for deployment.

### 3. Key Decisions
- Chose FastAPI for simplicity, automatic Swagger docs, and speed.
- Opted for Poetry for clean dependency management.
- Chose StratifiedKFold and macro F1-score to handle class balance.
- Used TF-IDF for classical models and BERT embeddings for advanced models.
- Used external storage (e.g., Google Drive) for model weights to keep the repo small.

#### On Hyperparameter Tuning
Due to strict time constraints and focus on demonstrating a complete end-to-end pipeline, we opted not to perform hyperparameter tuning. Instead, we selected well-known, reasonable defaults and focused on model comparison, explainability, and deployment. This tradeoff allowed for faster iteration while maintaining meaningful model evaluation. In a production or research setting, tuning would be essential to optimize model performance.

### 4. Deployment
- Dockerfile builds the API service and optionally downloads the model from an external source.
- Devcontainer support added for easy VSCode integration.
- API endpoints tested with Pytest and integrated with Swagger for doc visibility.
- MkDocs with mkdocstrings used to generate static API documentation.

---

## Implementation Steps

### 1. Data Analysis and Visualization
- Cleaned raw HTML (`<br />`) and lowercased reviews.
- Visualized label distributions, word clouds, n-grams.
- Analyzed lexical features, outliers, and sentiment trends.
- Used tools like spaCy, TextBlob, and Gensim for NER, polarity scoring, and topic modeling.

### 2. Text Vectorization
- Used TF-IDF for classical models and tokenizer-based encoding for deep models.
- TF-IDF helped interpret top discriminative tokens using `.feature_log_prob_` or `.feature_importances_`.

### 3. Model Development
- Trained classical models (LogisticRegression, NaiveBayes, SVM) using StratifiedKFold.
- Built and trained deep learning models (LSTM, BERT).
- Saved best models using joblib or HuggingFace’s save API.

### 4. Evaluation and Comparison
- Metrics: Accuracy, F1 macro, ROC AUC.
- Visualizations: Confusion matrix, ROC curve, PR curve, SHAP summary plots.
- Compared baseline TF-IDF + LogisticRegression against LSTM and BERT.
- Interpreted models using SHAP if possible, `.feature_importances_`, and `.feature_log_prob_` (NB).

### 5. API Implementation
- FastAPI with POST `/predict` endpoint.
- Returns label and confidence.
- Swagger + ReDoc auto-docs.
- Added test coverage with pytest + test client.

### 6. Containerization and Dev Support
- Used Docker for deployment and VSCode devcontainers.
- Managed Python dependencies via Poetry.
- Downloaded model from external source if `.env` provides link.

### 7. Documentation
- Generated API docs with FastAPI.
- Generated static project docs with MkDocs + mkdocstrings.
- Instructions provided for CLI/API usage, testing, and deployment.

---

## Run the Model API with Docker

### Requirements

* **Trained Models**
  1. At least **2GB RAM** for Docker container
  2. `models.zip` file contains trained classification model package file, downloaded from **privately provided link** (not publicly exposed for security reason.)
  3. Unzip the `models.zip` file at the project root directory. It should result the following project directory structure.

```
imdb-sentiment/
├── data/raw/
│       └── IMDB Dataset.csv
├── notebook/
│   ├── 01 Data Analysis.ipynb
│   ├── 02 Classical Models.ipynb
│   └── 03 Deep Models.ipynb
├── models/
│   ├── bert_finetuned/
│   ├── linearsvc.pkl
│   ├── logisticregression.pkl
│   ├── lstm_model.pt
│   ├── multinomialnb.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── apps/
│   │   ├── main.py
│   │   └── model_utils.py
│   └── tests/
│       ├── test_live_api.py
│       ├── test_main.py
│       └── test_model_utils.py
├── .gitignore
├── requirements.txt
├── Dockerfile
├── README.md
```

* **Otherwise**, the models need to be trained from scratch
  1. At least **16GB RAM** for Docker container and model training
  2. Create empty `models` folder
  3. Execute Jupyter Notebooks in `notebook` folder from `01-03` in order, which may takes about 14 hours when fine-tuning BERT model.
  4. Then rerun the container to try serving the BERT model again.

### Build Image
```bash
docker build -t imdb-sentiment .
```
### Run Container
```bash
docker run -it --rm -v "$(pwd):/app" -w /app -p 8000:8000 -p 8888:8888 imdb-sentiment
```

## How to Use the API

### Endpoint: `POST /predict`
Send a review as input JSON:
```json
{
  "review": "The movie was absolutely fantastic!"
}
```
Response:
```json
{
  "label": "positive",
  "confidence": 0.9975
}
```

### Example via Command Line
Use Unix `curl` or Windows `Invoke-RestMethod` to send a POST request:
- Bash
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"review": "The movie was absolutely fantastic!"}'
```
- Windows Powershell
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
      -Method POST `
      -Headers @{"Content-Type"="application/json"} `
      -Body '{"review": "The movie was absolutely fantastic!"}'
```

### API Documentation
Interactive documentation is automatically generated by FastAPI:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

Static Python API generated by MkDocs:
- Open HTML file [site/index.html](site/index.html) in local browser

## Testing

### Run All Test from Local (ensure server is running)
```bash
docker run -it --rm -v "$(pwd):/app" imdb-sentiment pytest
```

---

## Detecting Model Degradation & Auto-Redeployment Strategy

### Strategy for Detecting Degradation

We propose a multi-layered monitoring strategy:

#### 1. **Prediction Logging**

* Log all API inputs, outputs, and timestamps (optionally in a database or structured logs).
* Example:

  ```json
  {
    "timestamp": "2025-07-21T14:30:00Z",
    "input": "This movie was awful",
    "prediction": "positive",
    "confidence": 0.52
  }
  ```

#### 2. **Monitor Data Drift**

* Use tools like **Evidently AI** or custom checks to monitor:

  * Input text distribution changes (token frequency, length, sentiment polarity).
  * Model confidence score variance.
* Trigger warnings if drift exceeds a threshold.

#### 3. **Track Ground Truth (Optional)**

* If true labels are available later (e.g., user feedback), compare predictions and compute online metrics (accuracy, F1).

#### 4. **Health Dashboard**

* Visualize drift metrics, prediction histograms, and alerts using tools like Prometheus + Grafana or Streamlit.

---

### Strategy for Auto-Retraining & Redeployment

Once degradation is detected:

#### Step 1: **Trigger Model Retraining**

* Periodically or conditionally (e.g., drift > threshold) retrain the model on:

  * Most recent collected data + original training data.
* Run model evaluation scripts to compare metrics.

#### Step 2: **Model Versioning & Registry**

* Save new model artifacts to a registry (e.g., MLflow, Hugging Face Hub).
* Include metadata: training date, dataset stats, evaluation scores.

#### Step 3: **Automated Deployment**

* Use CI/CD pipelines (GitHub Actions, GitLab CI) to:

  * Build a new Docker image with updated model.
  * Deploy to staging → production via container orchestration (e.g., Docker Compose, Kubernetes).

---

## Notes
- This API uses a single trained model.
- You can switch or extend the model by editing `model_utils.py`.
- The current model expects preprocessed input compatible with DistilBERT tokenizer.