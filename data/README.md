# Data

This project uses the **Give Me Some Credit** dataset from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit

The raw data is **not included in this repo** (Kaggle's terms, and file size). To reproduce the analysis:

1. Sign in to Kaggle and open the competition page above.
2. Click **Join Competition** and accept the rules. You do not have to submit any predictions.
3. On the Data tab, download `cs-training.csv` and place it in this folder as `data/cs-training.csv`.

Files created by the notebooks (also not committed): `cs-training-clean.csv` (Phase 1), `train.csv` and `test.csv` (Phase 3).

## Columns

| Column | Meaning |
|---|---|
| `SeriousDlqin2yrs` | **Target.** 1 = 90+ days past due within 2 years, 0 = otherwise |
| `RevolvingUtilizationOfUnsecuredLines` | Credit card and credit line balances divided by credit limits |
| `age` | Age of borrower in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30-59 days late in the last 2 years |
| `DebtRatio` | Monthly debt payments and living costs divided by monthly gross income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Open loans (car, mortgage) and lines of credit (cards) |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Mortgages and real-estate loans, including home-equity lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60-89 days late in the last 2 years |
| `NumberOfDependents` | Family members supported, excluding the borrower |

Source of the definitions: the data dictionary provided on the Kaggle competition's Data tab.
