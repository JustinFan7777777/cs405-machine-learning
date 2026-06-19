# LAB06 Report: Linear Discriminant Analysis

## 1. Assignment Overview

This lab focuses on Linear Discriminant Analysis (LDA) in two parts:

1. Implementing LDA from scratch with NumPy to understand the mathematical mechanism.
2. Using Scikit-Learn LDA to classify handwritten digits on MNIST and evaluate performance.

The overall goal is to understand both:
- the theory of supervised dimensionality reduction,
- and its practical use in a real classification task.

---

## 2. Core Methods

## 2.1 Exercise 1: LDA From Scratch (NumPy)

The custom LDA implementation follows the standard pipeline:

1. Compute within-class scatter matrix:

$$
S_W = \sum_{k=1}^{C}\sum_{x \in D_k}(x-\mu_k)(x-\mu_k)^T
$$

It measures how spread out samples are inside each class.

2. Compute between-class scatter matrix:

$$
S_B = \sum_{k=1}^{C} n_k(\mu_k-\mu)(\mu_k-\mu)^T
$$

It measures how far class centers are from the global mean.

3. Solve generalized eigenvalue problem via:

$$
S_W^{-1}S_B
$$

and obtain eigenvalues/eigenvectors.

4. Sort eigenvalues descending and keep top components.

5. Project data:

$$
X_{proj} = XW
$$

where W is formed by selected eigenvectors.

Key implementation details:
- Maximum useful LDA components are limited by:

$$
\min(n_{features},\; C-1)
$$

- In practice, S_W may be singular. A stable implementation should use pseudo-inverse (pinv) instead of strict inverse.

---

## 2.2 Exercise 1 Visualization (Iris)

Iris has 3 classes, so LDA can produce at most 2 discriminant axes.

To keep a 3D plot style while respecting theory, the visualization uses:
- LD1 and LD2 as real discriminant axes,
- a padded zero auxiliary axis as the third axis.

This keeps plotting code readable while avoiding dimension mismatch errors.

---

## 2.3 Exercise 2: MNIST Classification with Scikit-Learn LDA

Pipeline used in the notebook:

1. Load MNIST (70000 samples, 784 features, 10 classes).
2. Standardize features with StandardScaler.
3. Split into training and testing sets.
4. Train LinearDiscriminantAnalysis.
5. Evaluate with:
- Accuracy
- Classification report (precision/recall/F1)
- Confusion matrix
- Train-test accuracy gap (overfitting check)

Interpretation of training logs:
- Model fitted successfully: normal.
- Number of classes = 10 and classes from 0 to 9: correct.
- Number of components used: None is normal when n_components=None is chosen (auto behavior).
- Projection matrix shape shown from coef_ = (10, 784): this is classifier parameter shape, not the transformed feature dimension.

---

## 3. Main Findings and Conclusions

1. LDA is a supervised dimensionality reduction method, so it uses labels and is usually more classification-oriented than unsupervised methods.
2. For low-dimensional, class-structured data (like Iris), LDA gives clear separation and interpretable axes.
3. For high-dimensional data (MNIST), LDA provides a fast and strong baseline classifier.
4. Correct interpretation of model attributes is important:
- coef_ shape does not mean output reduced dimension.
- n_components=None means automatic strategy, not failure.
5. Final model quality should be judged by test metrics and confusion matrix, not by a single printed attribute.

---

## 4. Q1: Advantages and Disadvantages of LDA

### Advantages

1. Supervised and discriminative:
- Uses class labels directly.
- Optimizes class separability.

2. Interpretable:
- Objective is explicit: maximize between-class separation and minimize within-class spread.

3. Efficient:
- Fast training and inference for many practical datasets.

4. Dual use:
- Can do both dimensionality reduction and classification.

### Disadvantages

1. Strong assumptions:
- Classical LDA assumes approximately Gaussian class distributions and similar class covariances.

2. Linear boundary limitation:
- May underfit highly nonlinear class structures.

3. Sensitivity:
- Can be sensitive to outliers and noisy features.

4. Component limit:
- Maximum reduced dimensions are constrained by C-1.

5. Numerical issues in custom implementation:
- S_W may be singular in high-dimensional or small-sample settings.

---

## 5. Q2: Difference Between LDA and PCA

1. Supervision
- LDA: supervised (uses labels).
- PCA: unsupervised (ignores labels).

2. Optimization target
- LDA: maximize class separability.
- PCA: maximize global variance preservation.

3. Dimensionality upper bound
- LDA: at most C-1.
- PCA: up to feature-space rank constraints, generally more flexible.

4. Typical use case
- LDA: classification-focused tasks.
- PCA: compression, denoising, visualization, generic preprocessing.

5. Effect on classification
- LDA often improves separability for labeled class problems.
- PCA may preserve variance that is not discriminative for labels.

---

## 6. Final Summary

This lab demonstrates that LDA is both theoretically meaningful and practically effective.

- In Exercise 1, implementing LDA from scratch clarifies how scatter matrices and eigen decomposition drive supervised projection.
- In Exercise 2, applying Scikit-Learn LDA to MNIST shows a complete machine learning workflow from preprocessing to evaluation.

Overall, LDA is a strong baseline when labels are available and class discrimination is the primary objective.