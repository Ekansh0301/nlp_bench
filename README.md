# Hindi Word Order Prediction
This project explores syntactic variation in Hindi sentences using features inspired by cognitive processing theories, based on the paper:

    Ranjan et al. (2022) – Locality and expectation effects in Hindi preverbal constituent ordering

---

##  Features Used

* **Trigram Surprisal**
  Measures how unexpected a word is given the two words before it (based on a trigram language model).

* **Dependency Length**
  Sum of distances between heads and dependents in the dependency parse; shorter is generally easier to process.

* **Information Status**
  Whether a noun phrase refers to given (old) or new information in the discourse.

* **Case Marker Transition Score**
  Captures patterns in case marking (e.g., subject/object markers) and their transitions between constituents.

* **PLM Score (Positional Language Model)**
  Probability of a word given its position in the sentence (useful for modeling Hindi's flexible word order).

* **UID (Uniform Information Density) Score** (to be uploaded)
  Measures how evenly information is distributed across a sentence, based on per-word surprisal.

---


##  Folder Overview

### `CODEX/LR/`

* Reimplementation of the original paper using **pairwise classification**.
* Each example is a pair of sentences (reference vs. variant).
* Uses logistic regression with and without interaction features.

### `CODEX/IMBALANCE/`

* Directly classifies **individual sentences** as reference or variant (no pairing).
* Focuses on handling class imbalance using different techniques.

### `CODEX/BALANCED_1/`

* Builds a **balanced dataset** by selecting 1 reference and 1 random variant per group.
* Trains and evaluates logistic regression on this smaller, balanced set.

### `DATA/`

* Contains feature files like surprisal scores, dependency lengths, and UID-based scores.

### `RESULTS/`

* Stores predictions, plots, and model performance summaries from different runs.

---