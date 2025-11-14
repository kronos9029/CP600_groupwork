# Sampling Report

- Total records: 32561
- Sample size: 5000
- Membership MI (bits): 0.3971
- Non-empty strata: 294

## Model Performance (Full vs Sample)
| Metric | Full Data | Sample |
| --- | --- | --- |
| accuracy | 0.804 | 0.805 |
| precision | 0.674 | 0.711 |
| recall | 0.362 | 0.320 |
| f1 | 0.471 | 0.441 |
| roc_auc | 0.827 | 0.824 |

## Distribution Drift (Total Variation Distance)
       feature  total_variation
     workclass         0.001200
    occupation         0.001382
           age         0.029861
  capital_gain         0.052252
        fnlwgt         0.107527
  capital_loss         0.111725
hours_per_week         0.197015

## Strata Allocation (top 15 by size)
                            stratum  group_size  allocated_sample
     (36-50, Private, Craft-repair)        1005               193
  (36-50, Private, Exec-managerial)         932               179
    (17-25, Private, Other-service)         833               160
     (26-35, Private, Craft-repair)         808               155
   (36-50, Private, Prof-specialty)         760               146
            (17-25, Private, Sales)         718               138
     (36-50, Private, Adm-clerical)         672               129
     (17-25, Private, Adm-clerical)         642               123
     (26-35, Private, Adm-clerical)         642               123
            (36-50, Private, Sales)         640               123
            (26-35, Private, Sales)         620               119
  (26-35, Private, Exec-managerial)         605               116
   (26-35, Private, Prof-specialty)         580               111
(36-50, Private, Machine-op-inspct)         571               110
    (26-35, Private, Other-service)         553               106

## Complexity Snapshot
| Component | Complexity |
| --- | --- |
| Overall | O(n*d) |
| Discretization | O(n*d) |
| Grouping | O(n) |
| Sampling | O(m*d) |
| Space | O(n + m) |

## Sample Classification Report
```
              precision    recall  f1-score   support

       <=50K      0.816     0.959     0.882      4945
        >50K      0.711     0.320     0.441      1568

    accuracy                          0.805      6513
   macro avg      0.763     0.639     0.661      6513
weighted avg      0.791     0.805     0.776      6513

```