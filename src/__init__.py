"""BNPL credit scorecard source package.

End-to-end pipeline:
    preprocess  - data cleaning, good/bad definition, train/test/oot split
    woe         - chi-square binning, WoE/IV, feature selection
    scorecard   - logistic regression + PDO scaling
    challenger  - XGBoost comparison
    validate    - KS/AUC/Gini/PSI/calibration and plots
"""

__version__ = "1.0.0"
