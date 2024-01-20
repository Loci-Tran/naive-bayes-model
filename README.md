# naive-bayes-model
# Patient Prescription Prediction Model

## Overview

This model predicts whether a patient will take their medicine (`C=1`) or not (`C=0`) based on a vector `X` representing the patient's current health status. The vector `X = [x_1, x_2, x_3, x_4]^T` consists of input figures corresponding to different health indicators:

- `x_1`: Weight Loss
- `x_2`: Headache
- `x_3`: Fever
- `x_4`: Cough

## Probabilities

- **Posterior Probability** $P(C_i|X)$: The probability of taking action $C_i$ given the health status vector $X$.
- **Likelihood** $P(X|C_i)$: The probability of observing the health status vector $X$ given action $C_i$.
- **Prior Probability** $P(C_i)$: The probability of action $C_i$ occurring in the dataset.
- **Marginal Probability** $P(X)$: The probability of observing the health status vector $X$ in the dataset.

## Bayes's Rule
The model uses Bayes's rule to calculate the probability of a patient taking their medicine:
$$P(C_i | X) = \frac{P(X | C_i) P(C_i)}{P(X)} = \frac{P(X | C_i) P(C_i)}{\sum P(X | C_i) P(C_i)}$$


## Reference

The methodology is based on concepts from the book *Introduction to Machine Learning* by Ethem Alpaydin.

---

**Note:** This model is for educational purposes and should not be used as a substitute for professional medical advice.
