# Sampling Report

- Total records: 32561
- Sample size: 5000
- Membership MI (bits): 0.3973
- Non-empty strata: 300

## Model Performance (Full vs Sample)
| Metric | Full Data | Sample |
| --- | --- | --- |
| accuracy | 0.804 | 0.804 |
| precision | 0.672 | 0.728 |
| recall | 0.363 | 0.294 |
| f1 | 0.471 | 0.419 |
| roc_auc | 0.834 | 0.829 |

## Distribution Drift (Total Variation Distance)
       feature  total_variation
     workclass         0.000986
    occupation         0.001514
           age         0.030903
  capital_gain         0.053737
        fnlwgt         0.107095
  capital_loss         0.110293
hours_per_week         0.192972

## Strata Allocation (top 15 by size)
                            stratum  group_size  allocated_sample
     (36-50, Private, Craft-repair)         991               190
  (36-50, Private, Exec-managerial)         939               180
    (17-25, Private, Other-service)         836               161
     (26-35, Private, Craft-repair)         785               151
   (36-50, Private, Prof-specialty)         769               148
            (17-25, Private, Sales)         700               134
     (36-50, Private, Adm-clerical)         658               126
            (36-50, Private, Sales)         644               124
     (17-25, Private, Adm-clerical)         630               121
     (26-35, Private, Adm-clerical)         620               119
            (26-35, Private, Sales)         619               119
  (26-35, Private, Exec-managerial)         598               115
   (26-35, Private, Prof-specialty)         571               110
(36-50, Private, Machine-op-inspct)         558               107
          (17-25, Unknown, Unknown)         540               104

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

       <=50K      0.812     0.965     0.882      4945
        >50K      0.728     0.294     0.419      1568

    accuracy                          0.804      6513
   macro avg      0.770     0.630     0.650      6513
weighted avg      0.792     0.804     0.770      6513

```