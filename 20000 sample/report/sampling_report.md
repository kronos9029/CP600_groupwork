# Sampling Report

- Total records: 32561
- Sample size: 20000
- Membership MI (bits): 0.2593
- Non-empty strata: 380

## Model Performance (Full vs Sample)
| Metric | Full Data | Sample |
| --- | --- | --- |
| accuracy | 0.804 | 0.807 |
| precision | 0.674 | 0.692 |
| recall | 0.362 | 0.355 |
| f1 | 0.471 | 0.469 |
| roc_auc | 0.827 | 0.826 |

## Distribution Drift (Total Variation Distance)
       feature  total_variation
    occupation         0.000374
     workclass         0.000482
           age         0.005165
  capital_gain         0.006702
  capital_loss         0.013075
        fnlwgt         0.023594
hours_per_week         0.068015

## Strata Allocation (top 15 by size)
                            stratum  group_size  allocated_sample
     (36-50, Private, Craft-repair)        1005               772
  (36-50, Private, Exec-managerial)         932               716
    (17-25, Private, Other-service)         833               640
     (26-35, Private, Craft-repair)         808               620
   (36-50, Private, Prof-specialty)         760               584
            (17-25, Private, Sales)         718               551
     (36-50, Private, Adm-clerical)         672               516
     (17-25, Private, Adm-clerical)         642               493
     (26-35, Private, Adm-clerical)         642               493
            (36-50, Private, Sales)         640               491
            (26-35, Private, Sales)         620               476
  (26-35, Private, Exec-managerial)         605               464
   (26-35, Private, Prof-specialty)         580               445
(36-50, Private, Machine-op-inspct)         571               438
    (26-35, Private, Other-service)         553               425

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

       <=50K      0.823     0.950     0.882      4945
        >50K      0.692     0.355     0.469      1568

    accuracy                          0.807      6513
   macro avg      0.757     0.652     0.675      6513
weighted avg      0.791     0.807     0.782      6513

```