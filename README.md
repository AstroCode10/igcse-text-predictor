# IGCSE English Language A (Edexcel IGCSE) – 2026 Text Predictor

A machine learning ranking model designed to estimate the probability of specific **non-fiction (Paper 1)** and **literary (Paper 2)** texts appearing in the **May/June 2026 Edexcel IGCSE English Language A examinations**.

Rather than treating prediction as a binary classification task, this project models exam selection as a **ranking problem**, capturing deeper structural patterns in how texts are rotated and selected across sessions.

---

## 📊 Model Performance

The model was optimized to balance **predictive accuracy** with **cross-validation stability**, ensuring robustness across limited historical data.

### Paper 1 (Non-Fiction)

| Metric | Value |
|------|------|
| **Best NDCG Score** | 0.5947 |
| **Stability (Std Dev)** | 0.0729 |

### Paper 2 (Literature)

| Metric | Value |
|------|------|
| **Best NDCG Score** | 0.6423 |
| **Stability (Std Dev)** | 0.0927 |

---

## 🎯 Key Findings

- **Text recurrence is strongly time-dependent**, with `Sessions_Since_Last` emerging as the most influential feature.
- **Paper 2 is more predictable than Paper 1**, suggesting tighter structural patterns in literary text selection.
- **Theme-based signals contribute secondary predictive power**, indicating that examiners balance thematic diversity.
- The model demonstrates that exam selection is **not random**, but influenced by:
  - Recency  
  - Frequency  
  - Thematic rotation  
  - Cross-stream usage patterns  

---

## 🧠 Methodology

This project uses a **learning-to-rank approach** via LightGBM.

Instead of predicting whether a text will appear, the model:

> **Ranks candidate texts based on their likelihood of appearing in the next exam session.**

### Target Definition

- **Label:** `App_Next` (binary)  
- Indicates whether a text appears in the **next session** for a given stream  

This transforms the problem into:

> “Given past data, rank texts by probability of future appearance.”

---

## ⚙️ Model Architecture

**Algorithm:** `LightGBM Ranker (LGBMRanker)`

Chosen for:
- Strong performance on structured/tabular data  
- Native support for ranking objectives  
- Ability to control overfitting on small datasets  

### Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|------|--------|
| `max_depth` | 3 | Prevents overfitting |
| `learning_rate` | 0.005 | Ensures gradual learning |
| `n_estimators` | 1500 | Allows convergence at low learning rate |
| `extra_trees` | True | Adds randomness for generalisation |
| `reg_alpha` | 0.5–0.7 | L1 regularisation |
| `reg_lambda` | 30 | Strong L2 stabilisation |
| `min_child_samples` | 35–55 | Prevents noisy splits |
| `subsample` | 0.55–0.65 | Reduces variance |
| `colsample_bytree` | 0.5–0.75 | Feature subsampling |

---

## 🔑 Feature Engineering

### Paper 1 Features

- `Num_Stream_App`  
- `Sessions_Since_Last`  
- `Theme_Gap`  
- `Theme_Likelihood`  
- `Text_Strength`  
- `Is_Overdue`  
- `Global_Sessions_Since_Last`  
- `Stream_App_Ratio`  
- `Paper_Stream_Non-R`, `Paper_Stream_R`  
- Text types: `Article`, `Memoir`, `Speech`, `Travelogue`  
- Themes: `Conflict`, `Personal`, `Social`  
- Text age: `Mid`, `Modern`  

---

### Paper 2 Features

- `Num_Stream_App`  
- `Sessions_Since_Last`  
- `Theme_Gap`  
- `Theme_Likelihood`  
- `Text_Strength`  
- `Is_Overdue`  
- `Global_Sessions_Since_Last`  
- `Stream_App_Ratio`  
- Text types: `Poem`, `Story`  
- Themes: `Personal`, `Resilience`, `Social`, `Suspense`  
- Text age: `Old`, `Modern`  

---

## 📈 Predictions (May/June 2026)

### Paper 1

#### Non-Regional Stream (Top 5)

| Rank | Text | Score |
|------|------|------|
| 1 | The Explorer’s Daughter | 0.0096 |
| 2 | Chinese Cinderella | -0.0287 |
| 3 | A Passage to Africa | -0.0928 |
| 4 | H is for Hawk | -0.0942 |
| 5 | A Game of Polo with a Headless Goat | -0.1953 |

#### Regional Stream (Top 5)

| Rank | Text | Score |
|------|------|------|
| 1 | The Danger of a Single Story | 0.2742 |
| 2 | A Game of Polo with a Headless Goat | -0.2211 |
| 3 | A Passage to Africa | -0.2261 |
| 4 | H is for Hawk | -0.2327 |
| 5 | Chinese Cinderella | -0.2390 |

---

### Paper 2

#### Non-Regional Stream (Top 5)

| Rank | Text | Score |
|------|------|------|
| 1 | Significant Cigarettes | -0.3520 |
| 2 | Whistle and I'll Come to You | -0.3583 |
| 3 | The Story of an Hour | -0.3682 |
| 4 | "Out, Out -" | -0.3781 |
| 5 | Still I Rise | -0.4172 |

#### Regional Stream (Top 5)

| Rank | Text | Score |
|------|------|------|
| 1 | "Out, Out -" | -0.3583 |
| 2 | Significant Cigarettes | -0.3718 |
| 3 | Whistle and I'll Come to You | -0.3781 |
| 4 | The Story of an Hour | -0.4253 |
| 5 | An Unknown Girl | -0.4766 |

---

## 🗂 Dataset

- **Total Rows per Paper:** 321  
- **Sessions per Paper:** 16  
- **Texts per Paper:** 10  

Each row represents a:

> **(Text, Session, Stream) candidate pairing**

with engineered features capturing:
- Historical frequency  
- Recency  
- Thematic trends  
- Cross-stream behaviour  

---

## 📈 Evaluation Method

The model uses **Normalized Discounted Cumulative Gain (NDCG)** to evaluate ranking quality.

### Why NDCG?

- Rewards correct ranking of top predictions  
- Penalizes misordering of high-relevance items  
- Ideal for ranking problems with binary relevance  

### Validation Strategy

**Time-Series Cross-Validation (TSCV)**

- Ensures only **past data predicts future sessions**  
- Prevents data leakage  
- Mimics real-world prediction conditions  

---

## ⚠️ Limitations

- Small dataset (16 sessions) limits generalisation  
- Relies purely on historical patterns  
- Cannot account for:
  - Examiner discretion  
  - Curriculum shifts  
  - Introduction of new texts  

Predictions should be interpreted as:

> **probabilistic signals, not guarantees**

---

## 📄 License

This project is provided for **educational and research purposes only**.

It is not affiliated with or endorsed by **Pearson Edexcel**.

---

## 👤 Author

Developed as an experimental machine learning project exploring predictive modelling on educational assessment patterns.

## 🚀 Installation

```bash
git clone https://github.com/yourusername/igcse-english-text-predictor.git
cd igcse-english-text-predictor
