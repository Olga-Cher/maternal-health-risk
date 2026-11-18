# Maternal health risk: predicting pregnancy risk with machine learning

A small end-to-end project. I checked the quality of a public dataset, trained two classification models to predict the risk level of pregnant women from six basic health measurements, and built a Power BI dashboard to show the results.

Tools: Python (pandas, scikit-learn) in Google Colab, Power BI.

![Dashboard](dashboard.png)

## Why this project

Many complications in pregnancy can be prevented or treated if they are found early. Where doctors are few, a simple tool that flags high-risk pregnancies from routine measurements could help decide who needs attention first.

The two possible errors have very different costs. Sending a healthy woman for an extra check-up takes some time. Missing a woman at high risk is far more serious. So the question I cared about most was how many high-risk women the model finds, and accuracy came second.

## Data

[Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data) (Kaggle, originally from the UCI Machine Learning Repository). The records come from hospitals and clinics in rural Bangladesh.

| Column | Meaning |
|---|---|
| Age | Age in years |
| SystolicBP / DiastolicBP | Blood pressure, mmHg |
| BS | Blood sugar, mmol/L |
| BodyTemp | Body temperature, °F |
| HeartRate | Heart rate, beats per minute |
| RiskLevel | Target: low / mid / high risk |

## Data quality findings

I checked the data before training anything, and this part changed the project more than the modelling did.

1. Two rows had a heart rate of 7 bpm, which is impossible for a living person. I removed every row below 40 bpm, which affected only those two.
2. 561 of the remaining 1,012 rows (55%) were exact duplicates. With duplicates left in, the same patient can land in both the training and the test set, and the model looks better than it is (data leakage). After removing them, 451 unique records were left.
3. About 20% of the original records (209 of 1,014) have an age under 16 or over 50, with values from 10 to 70. Some may be typing errors or women who were not pregnant. I kept them because I can't prove they are wrong, but they limit how far the results can be trusted.
4. 71 records (16% of the clean data) have exactly the same six measurements as another record but a different risk label. Most of these conflicts (24 of 35) are between low and mid risk. No model can tell apart patients who look identical in the data, so this caps how well any model can do.

Before moving into data analytics, I spent six years working on AI/ML data quality at Apple, evaluating model outputs and annotation consistency. Point 4 is a familiar problem from that work: when labels are not applied consistently, the model cannot learn a clear boundary.

After cleaning, the classes are imbalanced: 233 low risk, 112 high risk and 106 mid risk.

## Models and results

I split the data 80/20 into a training set (360 records) and a test set (91), keeping the same share of each risk level in both.

As a baseline, a "model" that always answers low risk would be right 52% of the time.

| | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.67 | 0.59 |
| Recall, high risk | 0.57 (13 of 23) | 0.70 (16 of 23) |
| Precision, high risk | 0.81 | 0.89 |
| Recall, mid risk | 0.05 | 0.14 |
| Recall, low risk | 1.00 | 0.74 |

Logistic Regression has the higher accuracy only because it predicts low risk for almost everyone, and it misses 10 of the 23 high-risk women. Random Forest finds 16 of them. When it flags a woman as high risk, it is right 89% of the time, so I chose it.

Neither model recognises mid risk. That fits the label conflicts above: in these six measurements, mid and low risk look almost the same.

Random Forest relies most on these features:

| Feature | Importance |
|---|---|
| Blood sugar (BS) | 0.37 |
| Systolic blood pressure | 0.18 |
| Age | 0.15 |
| Diastolic blood pressure | 0.11 |
| Heart rate | 0.11 |
| Body temperature | 0.09 |

Blood sugar matters about twice as much as anything else. In the test set, average blood sugar is 10.4 mmol/L for high-risk women, 7.4 for mid risk and 7.1 for low risk.

## Dashboard

The Power BI dashboard shows the Random Forest results on the test set. It has three summary numbers (patients in the test set, actual high-risk patients, high-risk patients found), the confusion matrix, average blood sugar by risk level, and a filter that shows only the model's 37 errors out of 91.

## Limitations

The test set is small: 91 patients, 23 of them high risk. Three patients more or less would move high-risk recall by 13 points, so a different split could give different numbers.

The questionable ages and conflicting labels also limit how much these results can be trusted.

This is not a clinical tool. The model shows which measurements go together with risk in this dataset, not what causes complications.

## Next steps

- Use cross-validation instead of one train/test split, for more stable numbers.
- Train only on ages 13 to 50 and compare with the full-data model.
- Push the model towards finding more high-risk cases, for example with class weights or a lower decision threshold.
- Merge mid and low risk into one class and test a simpler "high risk or not" model.

## Files

- `Maternal Health Risk Data Set.xls`: original dataset from Kaggle (a CSV file despite the .xls extension)
- `maternal_risk.ipynb`: data checks, cleaning, models and evaluation
- `test_predictions.csv`: test set with actual and predicted risk, used in Power BI
- `dashboard.png`: screenshot of the Power BI dashboard
