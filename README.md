# Book Chapter Recommendation System

## Problem
Given a user's reading history, predict the next chapter they should read.

## Approach
Hybrid recommender combining:
- Content-based filtering (genre similarity)
- Collaborative filtering (Jaccard user similarity)
- Sequential signals (reading progress)

## Project Structure
ASSIGNMENT/
├── data/              # Place chapters.csv and interactions.csv here
├── notebooks/         # Exploratory analysis
├── src/               # Core modules
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── collaborative_filter.py
│   ├── content_filter.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
└── main.py            # Entry point

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Add data files to data/ folder

3. Run:
   python main.py

## Results
- Hit Rate @ 5: ~47%
- MRR @ 5: ~0.13