# Sampling Report

- Total records: 32561
- Sample size: 10000
- Membership MI (bits): 0.4341
- Non-empty strata: 332

## Model Performance (Full vs Sample)
| Metric | Full Data | Sample |
| --- | --- | --- |
| accuracy | 0.804 | 0.805 |
| precision | 0.674 | 0.712 |
| recall | 0.362 | 0.320 |
| f1 | 0.471 | 0.442 |
| roc_auc | 0.827 | 0.825 |

## Distribution Drift (Total Variation Distance)
       feature  total_variation
    occupation         0.000548
     workclass         0.000754
           age         0.012578
  capital_gain         0.028252
  capital_loss         0.059225
        fnlwgt         0.071727
hours_per_week         0.161315

## Strata Allocation (top 15 by size)
                            stratum  group_size  allocated_sample
     (36-50, Private, Craft-repair)        1005               386
  (36-50, Private, Exec-managerial)         932               358
    (17-25, Private, Other-service)         833               320
     (26-35, Private, Craft-repair)         808               310
   (36-50, Private, Prof-specialty)         760               292
            (17-25, Private, Sales)         718               276
     (36-50, Private, Adm-clerical)         672               258
     (17-25, Private, Adm-clerical)         642               246
     (26-35, Private, Adm-clerical)         642               246
            (36-50, Private, Sales)         640               246
            (26-35, Private, Sales)         620               238
  (26-35, Private, Exec-managerial)         605               232
   (26-35, Private, Prof-specialty)         580               223
(36-50, Private, Machine-op-inspct)         571               219
    (26-35, Private, Other-service)         553               212

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
        >50K      0.712     0.320     0.442      1568

    accuracy                          0.805      6513
   macro avg      0.764     0.640     0.662      6513
weighted avg      0.791     0.805     0.776      6513

```