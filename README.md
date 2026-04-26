# 📚 Book Recommendation System

A hybrid recommendation system that predicts what a user should read next,
built on 1 million chapter-level reading interactions across 150,000 users
and 9,575 books.

---

## 🔗 Notebook with Full Output & Visualizations

> **[Click here to view the complete notebook with all outputs, charts, and results](https://github.com/KrishanKumar0805/Book_recommender/blob/main/notebooks/.ipynb_checkpoints/exploration-checkpoint.ipynb)**

This notebook contains:
- Full exploratory data analysis with charts
- Problem framing and assumptions
- Model architecture explanation
- Training outputs and evaluation results
- Demo recommendations for sample users
- Tradeoffs and future improvements

---

## 📁 Project 

Book_recommender/
├── data/
│   ├── chapters.csv          # 50,000 chapters across 9,575 books
│   └── interactions.csv      # 1,000,000 user-chapter interactions
├── models/
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb     # Main notebook (run this)
├── src/
│   ├── init.py
│   ├── data_loader.py        # Load and preprocess CSVs
│   ├── feature_engineering.py # Genre vectors for books and users
│   ├── collaborative_filter.py # ALS matrix factorization model
│   ├── content_filter.py     # Genre cosine similarity model
│   ├── hybrid_recommender.py # Combines CF + content + cold start
│   └── evaluation.py        # Leave-one-out split + HR@10 + MRR
├── main.py                   # Run full pipeline in one command
├── requirements.txt
└── README.md



---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/KrishanKumar0805/Book_recommender.git
cd Book_recommender
```

### 2. Place data files
Put `chapters.csv` and `interactions.csv` inside the `data/` folder.

### 3. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the full pipeline
```bash
python main.py
```

Expected output:
Loaded 50,000 chapters, 1,000,000 interactions
Users: 149,803 | Books: 9,575
Building book profiles...
Building user profiles...
Splitting train / test...
Fitting ALS model...
100%|████████| 20/20 [00:08<00:00]
ALS training complete.
Evaluating...
HR@10 : 0.0052
MRR   : 0.0016
Users : 148,548



Total runtime: ~3 minutes on a standard laptop.

---

## 🧠 Problem Framing

**Context:** Pratilipi is India's largest storytelling platform with 21M+ monthly
active users reading serialized stories across 12 Indian languages. The core
product challenge is keeping readers engaged by surfacing the right story at
the right time.

**Key data insight:** After analyzing the dataset, only **0.05% of users read
2+ chapters of the same book**. Users read across many different books rather
than reading a single book sequentially. This made "next chapter" recommendation
impossible — there is no sequential signal.

**Task defined as:** Given a user's chapter-reading history → recommend
10 new books they haven't interacted with yet.

---

## 🏗️ Model Architecture

### Two-Stage Hybrid Recommender

**Stage 1 — Candidate Generation:**

| Method | How it works |
|--------|-------------|
| **ALS Collaborative Filtering** | Matrix factorization on implicit user-book interactions |
| **Content-Based Filtering** | Genre cosine similarity between user profile and unseen books |

**Stage 2 — Score Blending:**

final_score = 0.7 × CF_score + 0.3 × Content_score


**Cold Start Strategy:**
| User history | Strategy |
|---|---|
| 0 interactions | Global popularity fallback |
| 1–2 interactions | Pure content-based (genre similarity) |
| 3+ interactions | Full hybrid |

---

## 📊 Evaluation

| Metric | Value |
|--------|-------|
| HR@10 | 0.0052 |
| MRR | 0.0016 |
| Random baseline HR@10 | 0.0010 |
| **Improvement over random** | **5x** |

**Why scores look low:** Leave-one-out on 9,575 books is an extremely
harsh metric. The model must identify 1 specific book out of 9,575 —
even a strong model scores low on this benchmark. The 5x improvement
over random confirms real collaborative signal was learned.

---

## ⚖️ Key Tradeoffs

| Decision | Reason |
|----------|--------|
| ALS over neural CF | Only 6.7 avg interactions/user — neural models would overfit |
| ALS over SVD | ALS handles implicit feedback correctly; SVD treats missing = dislike |
| Genre as content signal | Only rich metadata available; mirrors Pratilipi's category structure |
| α=0.7 favouring CF | Enough users for collaborative signal; tunable via grid search |
| Batch evaluation | Per-user loop took 10+ mins; batch ALS runs in under 60 seconds |

---

## 🚀 What I'd Improve with More Time

1. **Author affinity** — `author_id` exists in the data; users who read
   author X likely want more from author X
2. **Timestamps** — Add recency decay weighting; recent reads matter more
3. **Language filtering** — Critical for Pratilipi's 12-language platform;
   recommend within the user's language first
4. **Series detection** — Detect mid-series readers and prioritize
   next chapter of their current story
5. **Tune hyperparameters** — Grid search over ALS factors (64/128/256)
   and blending weight α
6. **Scale with Spark ALS** — For Pratilipi's 21M+ users, same algorithm
   implemented in Apache Spark MLlib handles full scale
7. **A/B testing** — Any production model change at Pratilipi would go
   through CTR and session-length A/B tests before full rollout

---

## 📦 Dependencies
pandas
numpy
scikit-learn
implicit
scipy
jupyter
ipykernel

Install with: `pip install -r requirements.txt`

---

## 👤 Author
Krishan Kumar