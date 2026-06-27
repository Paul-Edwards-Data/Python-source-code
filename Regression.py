"""
================================================================================
ATTRIBUTION & ACKNOWLEDGEMENT NOTICE
Statistical Analysis Framework: Post-Simulation Privacy & Adversarial Attack Evaluation
Author: Paul Edwards

References:
- Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain", arXiv, 2017
- Seabold, S., & Perktold, J. statsmodels: Econometric and statistical modeling with Python, 2010
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS, 2019
================================================================================
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

# --- 1. Data Ingestion & Preprocessing ---
# Load empirical simulation results matrix
data_path = "/content/sample_data/MNIST.csv"
if not os.path.exists(data_path):
    # Fallback to local working directory if executing outside of original environment
    data_path = "MNIST.csv"

df = pd.read_csv(data_path)
print(f"Dataset Loaded Successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

# Define binary outcome variable based on Attack Success Rate (ASR) threshold specification
df["AttackSuccess"] = (df["ASR"] > 0.50).astype(int)

# Feature Engineering: Account for non-linear, quadratic relationships in privacy budget metrics
df["Epsilon2"] = df["Epsilon"] ** 2

# Cast structural independent variables into categorical data types for dummy variable encoding
df["Algorithm"] = df["Algorithm"].astype("category")
df["IID"] = df["IID"].astype("category")

# --- 2. Logistic Regression Modeling ---
# Specify formal econometric formula testing interaction effects and non-linear boundaries
formula = """
AttackSuccess ~ 
    C(IID, Treatment(reference=False)) + 
    C(Algorithm, Treatment(reference='FedAvg')) + 
    Epsilon + 
    Epsilon2 + 
    Poison_Rate + 
    Epsilon:Poison_Rate + 
    Epsilon:C(IID, Treatment(reference=False))
"""

print("\nExecuting Maximum Likelihood Estimation (MLE) for Logit Model...")
model = smf.logit(formula=formula, data=df).fit(maxiter=100)
print(model.summary())

# --- 3. Statistical Metrics & Feature Odds Ratios ---
params = model.params
conf = model.conf_int()

results_table = pd.DataFrame({
    "Beta": params,
    "Std_Error": model.bse,
    "Wald_Chi2": (params / model.bse) ** 2,
    "p_value": model.pvalues,
    "Odds_Ratio": np.exp(params),
    "CI_Lower": np.exp(conf[0]),
    "CI_Upper": np.exp(conf[1])
}).round(4)

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION COMPREHENSIVE COEFFICIENTS")
print("=" * 80)
print(results_table)

# --- 4. Model Diagnostics & Omni-Health Checks ---
ll_model = model.llf
ll_null = model.llnull
lr_stat = 2 * (ll_model - ll_null)
df_lr = int(model.df_model)
lr_p = chi2.sf(lr_stat, df_lr)

diagnostics = pd.DataFrame({
    "Metric": [
        "N (Observations Count)",
        "Log-Likelihood (Fitted Model)",
        "Log-Likelihood (Null Model)",
        "McFadden Pseudo R-squared",
        "Likelihood Ratio Chi-Square statistic",
        "LR Test probability (p-value)",
        "Degrees of Freedom"
    ],
    "Value": [
        len(df),
        ll_model,
        ll_null,
        model.prsquared,
        lr_stat,
        lr_p,
        df_lr
    ]
})

print("\n" + "=" * 80)
print("MODEL FITNESS DIAGNOSTICS")
print("=" * 80)
print(diagnostics)

# --- 5. Data Exportation ---
results_table.to_csv("MNIST_Logistic_Regression_Table.csv", index=True)
diagnostics.to_csv("MNIST_Model_Diagnostics.csv", index=False)
print("\n[SUCCESS] Statistical matrices successfully exported to local CSV structures.")

# --- 6. Theoretical Boundary (Tipping Point) ---
# Calculate the critical mathematical vertex point where the quadratic Epsilon curve shifts direction
turning_point = -model.params["Epsilon"] / (2 * model.params["Epsilon2"])
print(f"\nCalculated Quadratic Inflection Tipping Point (Epsilon): {turning_point:.6f}")
