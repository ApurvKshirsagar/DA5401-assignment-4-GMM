# Assignment 4

## Name: Apurv Ravindra Kshirsagar

## Roll No: CE22B042

---

## Project Overview

This project explores **GMM-based synthetic sampling** for handling class imbalance in fraud detection. The dataset used is the **Credit Card Fraud Detection dataset** (Kaggle), which is highly imbalanced with fraudulent transactions forming less than 0.2% of all records.

The objective is to:

1. Build a **baseline Logistic Regression model** on the imbalanced dataset.
2. Implement **Gaussian Mixture Model (GMM)-based oversampling** to generate realistic synthetic samples for the minority class.
3. Combine **Clustering-Based Undersampling (CBU)** of the majority class with **GMM oversampling** of the minority class to form a balanced dataset.
4. Compare the performance of these models on the **minority (fraudulent) class** using Precision, Recall, and F1-score.
5. Conclude with **recommendations** on the effectiveness of GMM-based synthetic sampling.

---

## Steps Performed

### 1. Data Exploration and Baseline Model

- Loaded the **creditcard.csv** dataset.
- Dropped irrelevant features (`Time`) and standardized `Amount`.
- Analyzed class imbalance (fraud cases ≈ 0.17% of data).
- Trained **Logistic Regression on imbalanced data (Baseline Model)**.
  - **Observation**: Accuracy was misleadingly high (>99%) while recall was poor, confirming the limitations of accuracy in imbalanced settings.

### 2. GMM-Based Oversampling

- Fitted a **Gaussian Mixture Model** on the minority (fraud) class.
- Determined optimal number of components `k` using **AIC** and **BIC**.
- Generated synthetic fraud samples by sampling from the learned GMM distribution.
- Combined synthetic samples with original training data to create a balanced dataset.

### 3. CBU + GMM Hybrid Strategy

- Applied **Clustering-Based Undersampling (CBU)** to reduce the majority (non-fraud) class in a representative way.
- Performed **GMM-based oversampling** on the minority class to match the reduced majority size.
- Created a smaller but balanced training dataset.

### 4. Model Comparison and Analysis

- Constructed a **summary table and bar chart** comparing Baseline, GMM Oversampling, and CBU+GMM models.
- Key findings:
  - **Baseline** → High precision, but low recall (missed many fraud cases).
  - **GMM Oversampling** → Recall improved drastically (~90%), but precision dropped sharply (~8%).
  - **CBU + GMM** → Recall remained high (~89%), but precision dropped further (~3%).

### 5. Conclusions and Recommendations

- **Benefits**:

  - GMM-based oversampling significantly improves recall, making the model much better at detecting fraudulent transactions.
  - CBU ensures majority class diversity while balancing dataset size.

- **Drawbacks**:

  - Precision drops heavily in both oversampling strategies, leading to many false positives.
  - CBU+GMM reduces dataset size too aggressively, further lowering precision.

- **Best Performing Method**:

  - GMM oversampling alone gave the best trade-off (high recall, moderate F1-score compared to CBU+GMM).

- **Recommendation**:
  - For fraud detection, **GMM oversampling is effective when recall is prioritized** (catching as many frauds as possible).
  - In real-world deployment, GMM oversampling should be combined with **threshold tuning, cost-sensitive learning, or ensemble methods** to reduce false positives.

---

## Libraries Required

Install the following Python libraries before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
