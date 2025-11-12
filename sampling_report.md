# Sampling Report

- Total records: 32561
- Sample size: 26048 (full training set)
- Non-empty strata: 380

## Model Performance (Full vs Sample)
| Metric | Full Data | Sample |
| --- | --- | --- |
| accuracy | 0.804 | 0.804 |
| precision | 0.674 | 0.674 |
| recall | 0.362 | 0.362 |
| f1 | 0.471 | 0.471 |
| roc_auc | 0.827 | 0.827 |

## Distribution Drift (Total Variation Distance)
       feature  total_variation
           age              0.0
     workclass              0.0
        fnlwgt              0.0
    occupation              0.0
  capital_gain              0.0
  capital_loss              0.0
hours_per_week              0.0

## Strata Allocation (top 15 by size)
                            stratum  group_size  allocated_sample
     (36-50, Private, Craft-repair)        1005              1005
  (36-50, Private, Exec-managerial)         932               932
    (17-25, Private, Other-service)         833               833
     (26-35, Private, Craft-repair)         808               808
   (36-50, Private, Prof-specialty)         760               760
            (17-25, Private, Sales)         718               718
     (36-50, Private, Adm-clerical)         672               672
     (17-25, Private, Adm-clerical)         642               642
     (26-35, Private, Adm-clerical)         642               642
            (36-50, Private, Sales)         640               640
            (26-35, Private, Sales)         620               620
  (26-35, Private, Exec-managerial)         605               605
   (26-35, Private, Prof-specialty)         580               580
(36-50, Private, Machine-op-inspct)         571               571
    (26-35, Private, Other-service)         553               553

## Sample Classification Report
```
              precision    recall  f1-score   support

       <=50K      0.824     0.945     0.880      4945
        >50K      0.674     0.362     0.471      1568

    accuracy                          0.804      6513
   macro avg      0.749     0.653     0.675      6513
weighted avg      0.788     0.804     0.781      6513

```