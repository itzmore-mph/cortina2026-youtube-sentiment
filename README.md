# Cortina 2026 YouTube Sentiment Analysis

Sentiment analysis of YouTube comments about the **Winter Olympic Games Milano Cortina 2026**, using DistilBERT for labelling and custom RNN/Seq2Seq models for classification.

This is the submission notebook for **Module 6 - Telling the game: NLP for sports intelligence** of the MSc in Artificial Intelligence Applied to Sports (Sports Data Campus / UCAM / ENIIT).

---

## Project overview

The pipeline covers the full NLP workflow in six phases:

| Phase | Description |
|-------|-------------|
| 1 | YouTube comment collection via the Data API v3 |
| 2 | Sentiment labelling with DistilBERT (transfer learning) |
| 3 | Data cleaning and exploratory data analysis |
| 4 | Building a Simple RNN (GRU) and a Seq2Seq (Encoder-Decoder) model |
| 5 | Model training and hyperparameter grid search |
| 6 | Evaluation, confusion matrices, and critical reflection |

---

## Results

| Model | Test Accuracy |
|---|---|
| Majority-class baseline | 51.06% |
| Simple RNN (GRU) | 42.55% |
| Seq2Seq (Encoder-Decoder) | 51.06% |

Both models struggle on this dataset due to its small size (~215 training samples after cleaning) and noisy pseudo-labels from an English-only DistilBERT model applied to a multilingual comment set. The notebook includes a detailed critical reflection on these limitations and concrete proposed improvements.

---

## Dataset

- **Source:** YouTube Data API v3, single video (USA vs Canada ice hockey gold medal, Milano Cortina 2026)
- **Raw comments collected:** 500
- **After cleaning:** 308 (61.6%)
- **Class split:** 48.7% Positive / 51.3% Negative (near-balanced)
- **Languages:** English, Russian, Korean, Japanese, Spanish, Portuguese, and others

The raw and processed CSVs are not tracked in git (see `.gitignore`). Re-running the notebook regenerates them from the API.

---

## Repository structure

```
.
+-- olympics_nlp_sentiment.ipynb   # Main notebook (all 6 phases)
+-- requirements.txt               # Pinned dependencies
+-- .env.example                   # Template for API credentials
+-- data/
|   +-- .gitkeep                   # Placeholder (CSVs are gitignored)
+-- figures/                       # Generated plots (gitignored)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/itzmore-mph/cortina2026-youtube-sentiment.git
cd cortina2026-youtube-sentiment
```

### 2. Create and activate a Python 3.11 environment

```bash
conda create -n cortina2026 python=3.11
conda activate cortina2026
pip install -r requirements.txt
```

### 3. Configure credentials

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
YOUTUBE_API_KEY=your_youtube_data_api_v3_key
YOUTUBE_VIDEO_ID=fw6NOvuO67s
```

You need a [YouTube Data API v3 key](https://console.cloud.google.com/). The free quota (10,000 units/day) is sufficient for collecting 500 comments.

### 4. Run the notebook

Open `olympics_nlp_sentiment.ipynb` in Jupyter or VS Code and run all cells. The notebook is idempotent: if the data CSVs already exist it skips the API call and DistilBERT inference automatically.

---

## Reproducibility

All random seeds are fixed via a `set_seed(42)` helper that covers:
- NumPy
- PyTorch (CPU and CUDA)
- Python built-in `random`
- cuDNN deterministic mode
- DataLoader shuffle generator

Running the notebook twice produces identical outputs.

---

## Dependencies

| Package | Version |
|---|---|
| torch | 2.11.0 |
| transformers | 5.3.0 |
| google-api-python-client | 2.193.0 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| wordcloud | 1.9.6 |
| python-dotenv | 1.2.2 |

---

## License

For academic submission purposes only.
