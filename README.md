# IGCSE English Language A (Edexcel IGCSE) – 2026 Text Predictor

A machine learning ranking model designed to estimate the probability of specific **non-fiction texts** appearing in the **May/June 2026 Edexcel IGCSE English Language A examinations (Paper 1 and Paper 2).**

The model analyses historical exam patterns and ranks texts based on statistical likelihood using a stability-focused ranking architecture.

---

## 📊 Model Performance

The model was optimized to balance **predictive accuracy** with **cross-validation stability**, ensuring that predictions reflect long-term historical patterns rather than single-year anomalies.

| Metric | Value |
|------|------|
| **Best NDCG Score** | 0.5997 |
| **Stability (Std Dev)** | 0.1074 |
| **Core Algorithm** | LightGBM Ranker (`LGBMRanker`) |
| **Validation Strategy** | Time-Series Cross-Validation (TSCV) |

---

## 🧠 Methodology

The project uses a **learning-to-rank approach**, where historical exam data is treated as a ranking problem rather than simple classification.

Instead of predicting whether a text will appear, the model **ranks candidate texts by likelihood of appearance** based on historical exam patterns.

A **stability-first architecture** was used to reduce overfitting to individual exam sessions.

Key ideas:

- Historical rotation patterns influence text selection
- Thematic trends affect exam composition
- Text recurrence intervals provide predictive signal

---

## ⚙️ Model Architecture

**Algorithm:** `LightGBM Ranker`

LightGBM was selected due to its strong performance on structured tabular datasets and support for ranking objectives.

### Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|------|--------|
| `extra_trees` | True | Adds randomness to tree splits to reduce variance |
| `max_depth` | 3 | Restricts tree complexity to prevent overfitting |
| `min_child_samples` | 35 | Ensures each leaf node has sufficient data |
| `reg_alpha` | Moderate | L1 regularization to prune weak features |
| `reg_lambda` | Moderate | L2 regularization to stabilize the model |

---

## 🔑 Feature Importance (by Gain)

The following features were the most influential in determining the ranking predictions:

1. **Sessions_Since_Last**  
   Measures how many exam sessions have passed since the text last appeared.  
   This was the strongest predictor of recurrence probability.

2. **Num_Stream_App**  
   Tracks how frequently a text appears across exam streams  
   (e.g. Regional vs Non-Regional papers).

3. **Primary_Theme_Social**  
   A thematic indicator capturing whether the text belongs to commonly recurring social themes.

---

## 🗂 Dataset

The dataset was constructed using historical **Edexcel IGCSE English Language A exam papers**, including:

- Past Paper 1 texts
- Past Paper 2 texts
- Exam session metadata
- Theme classification
- Appearance frequency metrics

Each row represents a **text-session candidate pairing** with engineered features describing historical context.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/igcse-english-text-predictor.git
cd igcse-english-text-predictor
```
---

## 📈 Evaluation Method

The model is evaluated using **Normalized Discounted Cumulative Gain (NDCG)**, which measures how well the model ranks the most relevant texts near the top of the list.

Time-series cross-validation ensures that training only uses **past exam data** to predict future sessions, preventing information leakage.

---

## ⚠️ Limitations

- The model relies on historical patterns and cannot account for **human examiner discretion**.
- Predictions should be interpreted as **probabilistic guidance**, not certainty.
- Dataset size is limited due to the relatively small number of exam sessions available.

---

## 📄 License

This project is provided for **educational and research purposes only**.

It is not affiliated with or endorsed by **Pearson Edexcel**.

---

## 👤 Author

Developed as an experimental machine learning project exploring predictive modelling on educational assessment patterns.
