
# Ultra‑Sparse Matrix Factorization Recommender System
A study of Biased Latent Matrix Factorization (BLMF) performance under extreme sparsity

## Overview
This repository explores how well a Biased Latent Matrix Factorization (BLMF) recommender system performs when trained on extremely sparse rating matrices, using a reduced sample of the MovieLens 32M dataset.
The goal:

Determine the minimum viable data density needed for smaller companies to build effective recommendation systems without requiring large‑scale compute or deep learning architectures.

We compare two models:

Regularized Biased Baseline Recommender
Biased Latent Matrix Factorization (BLMF) implemented via Mini‑Batch SGD

Both are evaluated across 10 progressively sparser datasets while preserving matrix structure (no user/movie loses its final rating).

## Dataset
Source: MovieLens 32M (GroupLens Research).
Subset used:

150,000 ratings (≈0.0152% density of the full dataset)
75,679 users
13,054 movies

A custom sampling process ran 300 random seeds to find the densest possible 150k‑rating subset.

## Problem Motivation
Recommender Systems degrade when data becomes sparse. Smaller companies often lack high‑density datasets; thus this project answers:

How sparse can data become before MF‑based systems stop working?
Can a lightweight BLMF model outperform a simple Baseline under extreme sparsity?
What density is “good enough” for a viable recommender?


## Methodology
__1. Sparse Matrix Reduction Algorithm__  
A custom algorithm removes ratings while guaranteeing:  
✔ each user keeps ≥1 rating  
✔ each movie keeps ≥1 rating  
✔ matrix structure remains valid  
This allows controlled sparsity experiments.

_2. Models Compared_
Baseline Recommender (Regularized Bias Model)
Predicts ratings using:

Global mean
User bias
Movie bias
L2 regularization

Serves as a simple and interpretable benchmark.
Biased Latent Matrix Factorization (BLMF)
Matrix Factorization with:

Latent dimension K ∈ {5,10,20,30}
Mini‑Batch SGD optimization
Tunable learning rate + regularization
Dot‑product latent interaction terms

600 total model variants were trained across sparsity levels.

3. Evaluation Metrics
Accuracy Metrics:

RMSE
MAE

Effectiveness Metrics:

Precision@K
Recall@K

Because these metrics are on different scales, a custom synthesized score was developed to combine them fairly.

📊 Results Summary
✔️ BLMF generally outperforms Baseline
Across nearly all sparsity levels, BLMF achieves:

lower RMSE & MAE
higher Recall
better synthesized scores

✔️ One exception
At the absolute sparsest density (~0.000137%), Baseline slightly outperformed BLMF—but both models performed poorly.
✔️ At ultra‑sparse densities, Recall > Precision
Because most users had only 1–2 test ratings, Recall naturally inflated.
❌ Below density ≈0.00007
No model produced reliable recommendations.
Both accuracy and ranking collapsed.

🧪 Example Model Output
Even with 0.00152% density, the BLMF model successfully recommended movies that users actually rated in the held‑out test set (e.g., correctly recommending Shawshank Redemption for user 54820).

🌟 Key Insights

Lightweight MF models can outperform simple baselines even at extremely low densities.
But ultra‑sparse data severely limits usefulness, regardless of model.
For real‑world use, a recommender should enforce a minimum density or minimum user rating count.


🚧 Limitations

58% of users were single‑raters (standard deviation = 0).
5.88% rated all movies the same (also zero standard deviation).
Dataset limited to 150k ratings due to memory constraints.
Extremely sparse matrices can produce unstable behavior.


🚀 Future Work
Potential improvements include:

Scale tests to 500k–1M+ ratings
Explore implicit feedback (clicks, views)
Integrate hybrid models using metadata (genre, tags)
Establish practical density thresholds (e.g., density needed for ≥20% Precision/Recall)
Compare with modern deep‑learning recommenders
