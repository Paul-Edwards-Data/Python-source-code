# ==============================================================================
# ATTRIBUTION & ACKNOWLEDGEMENT NOTICE
# This analytical framework was written/adapted by Paul Edwards.
#
# The implementation incorporates standard machine learning workflows and
# publicly available algorithmic logic
# adapted from their respective original research publications.
# All mathematical concepts and libraries utilized are credited to their
# respective authors.
# References:
# Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain", arXiv, 2017
# Seabold, S., & Perktold, J. statsmodels: Econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference, 2010
# P. Virtanen and et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, pp. 261–272, 2020. 
# P. Blanchard, E. M. El Mhamdi, R. Guerraoui and J. Stainer, "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", Advances in Neural Information Processing Systems 30, vol. 30, pp. 119–129, 2019. 
# ==============================================================================

# Import in tools to work with data tables and maths
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2

# ============================================================
# Load dataset
# ============================================================

# Read the data file from the folder path
df = pd.read_csv("/content/sample_data/MNIST.csv")

# Print how many rows and columns are in the file
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# ============================================================
# Create binary outcome
# Dissertation specification:
# AttackSuccess = 1 when ASR > 0.50 and ASR > 0.20
# ============================================================
# df["AttackSuccess"] = (df["ASR"] > 0.20).astype(int)

# Make a new column named AttackSuccess. It sets the value to 1 if ASR is bigger than 0.50, and 0 if it is not.
df["AttackSuccess"] = (df["ASR"] > 0.50).astype(int)

# ============================================================
# Feature engineering
# ============================================================

# Create a new column by squaring the Epsilon numbers
df["Epsilon2"] = df["Epsilon"] ** 2

# Treat these two columns as categories instead of regular text or numbers
df["Algorithm"] = df["Algorithm"].astype("category")
df["IID"] = df["IID"].astype("category")

# ============================================================
# Fit logistic regression
# ============================================================

# The maths for the model. 
# It tests how different factors change the chance of an attack succeeding.
# It sets 'False' as the baseline for IID and 'FedAvg' as the baseline for Algorithm.
formula = """
AttackSuccess ~
C(IID, Treatment(reference=False))
+ C(Algorithm, Treatment(reference='FedAvg'))
+ Epsilon
+ Epsilon2
+ Poison_Rate
+ Epsilon:Poison_Rate
+ Epsilon:C(IID, Treatment(reference=False))
"""

# Build the logistic regression model using the recipe and the data table.
# Run the model up to 100 times to find the best fit.
model = smf.logit(
    formula=formula,
    data=df
).fit(maxiter=100)

# Print a big summary sheet of the model results
print(model.summary())

# ============================================================
# Odds ratios and confidence intervals
# ============================================================

# Find raw model scores and calculate the confidence intervals for fitted model parameters
params = model.params
conf = model.conf_int()

# Create a clean data table to store the final statisics
results_table = pd.DataFrame({
    "Beta": params,                                # The raw effect size
    "Std_Error": model.bse,                        # The margin of error for the score
    "Wald_Chi2": (params / model.bse) ** 2,        # A test score to see if the factor matters
    "p_value": model.pvalues,                      # The probability that an independent variable's effect is zero (random luck). 
    "Odds_Ratio": np.exp(params),                  # Changes the score into an odds ratio
    "CI_Lower": np.exp(conf[0]),                   # The low end of the odds ratio range
    "CI_Upper": np.exp(conf[1])                    # The high end of the odds ratio range
})

# Round all the numbers in the table to 4 decimal places
results_table = results_table.round(4)

# Print the final results table to the screen
print("\n")
print("=" * 80)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 80)
print(results_table)

# ============================================================
# Model diagnostics
# ============================================================

# Get accuracy scores for this model vs a blank model with no factors
ll_model = model.llf
ll_null = model.llnull

# Calculate a score to see how much better this model is than a blank one
lr_stat = 2 * (ll_model - ll_null)

# Get the number of factors used in the model
df_lr = int(model.df_model)

# Calculate the final safety check probability score (p-value)
lr_p = chi2.sf(lr_stat, df_lr)

# Put all the model health checks into a table
diagnostics = pd.DataFrame({
    "Metric": [
        "N",
        "Log-Likelihood (Model)",
        "Log-Likelihood (Null)",
        "McFadden Pseudo R²",
        "Likelihood Ratio χ²",
        "LR Test p-value",
        "Degrees of Freedom"
    ],
    "Value": [
        len(df),          # Total rows used
        ll_model,         # Model fit score
        ll_null,          # Blank model fit score
        model.prsquared,  # How much variation the model explains
        lr_stat,          # Overall model test statistic
        lr_p,             # Overall model p-value
        df_lr             # Count of predictors
    ]
})

# Print the health check table to the screen
print("\n")
print("=" * 80)
print("MODEL DIAGNOSTICS")
print("=" * 80)
print(diagnostics)

# ============================================================
# Export tables
# ============================================================

# Save the main results table as a spreadsheet file
results_table.to_csv(
    "MNIST_Logistic_Regression_Table.csv",
    index=True
)

# Save the health check table as a spreadsheet file
diagnostics.to_csv(
    "MNIST_Model_Diagnostics.csv",
    index=False
)

# Display a success message
print("\nTables exported successfully.")

# Calculate the tipping point
turning_point = -model.params["Epsilon"] / (
2 * model.params["Epsilon2"]
)

# Display that tipping point
print(turning_point)
