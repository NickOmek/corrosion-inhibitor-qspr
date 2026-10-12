| Features | Model | CV | R² | RMSE | MAE |
|---|---|---|---|---|---|
| Descriptors + ECFP4 | XGBoost | random | 0.65 ± 0.01 | 7.2 ± 0.1 | 5.0 ± 0.1 |
| Descriptors + ECFP4 | Random forest | random | 0.62 ± 0.01 | 7.5 ± 0.1 | 5.2 ± 0.1 |
| Descriptors + ECFP4 | Ridge | random | 0.51 ± 0.01 | 8.5 ± 0.1 | 6.1 ± 0.0 |
| Descriptors + ECFP4 | Neural network (MLP) | random | 0.47 ± 0.01 | 8.8 ± 0.1 | 6.4 ± 0.1 |
| Descriptors + ECFP4 | Mean baseline | random | -0.01 ± 0.00 | 12.2 ± 0.0 | 8.8 ± 0.0 |
| ECFP4 | XGBoost | random | 0.47 ± 0.02 | 8.8 ± 0.2 | 5.9 ± 0.1 |
| ECFP4 | Ridge | random | 0.43 ± 0.02 | 9.1 ± 0.1 | 6.5 ± 0.1 |
| ECFP4 | Random forest | random | 0.36 ± 0.02 | 9.7 ± 0.1 | 6.2 ± 0.1 |
| ECFP4 | Neural network (MLP) | random | 0.27 ± 0.05 | 10.4 ± 0.4 | 7.4 ± 0.3 |
| ECFP4 | Mean baseline | random | -0.01 ± 0.00 | 12.2 ± 0.0 | 8.8 ± 0.0 |
| RDKit descriptors | XGBoost | random | 0.65 ± 0.01 | 7.2 ± 0.1 | 5.0 ± 0.1 |
| RDKit descriptors | Random forest | random | 0.62 ± 0.01 | 7.5 ± 0.1 | 5.3 ± 0.1 |
| RDKit descriptors | Ridge | random | 0.45 ± 0.03 | 9.0 ± 0.2 | 6.6 ± 0.2 |
| RDKit descriptors | Neural network (MLP) | random | 0.33 ± 0.02 | 9.9 ± 0.2 | 7.3 ± 0.2 |
| RDKit descriptors | Mean baseline | random | -0.01 ± 0.00 | 12.2 ± 0.0 | 8.8 ± 0.0 |
| Descriptors + ECFP4 | XGBoost | scaffold | 0.52 ± 0.01 | 8.4 ± 0.0 | 6.0 ± 0.1 |
| Descriptors + ECFP4 | Random forest | scaffold | 0.51 ± 0.01 | 8.5 ± 0.1 | 6.0 ± 0.1 |
| Descriptors + ECFP4 | Ridge | scaffold | 0.33 ± 0.01 | 9.9 ± 0.1 | 7.4 ± 0.1 |
| Descriptors + ECFP4 | Neural network (MLP) | scaffold | 0.33 ± 0.05 | 9.9 ± 0.3 | 7.5 ± 0.4 |
| Descriptors + ECFP4 | Mean baseline | scaffold | -0.02 ± 0.00 | 12.2 ± 0.0 | 8.9 ± 0.0 |
| ECFP4 | Ridge | scaffold | 0.23 ± 0.01 | 10.6 ± 0.1 | 7.9 ± 0.1 |
| ECFP4 | XGBoost | scaffold | 0.19 ± 0.02 | 10.9 ± 0.1 | 7.4 ± 0.2 |
| ECFP4 | Random forest | scaffold | 0.12 ± 0.01 | 11.4 ± 0.1 | 7.4 ± 0.2 |
| ECFP4 | Neural network (MLP) | scaffold | 0.08 ± 0.07 | 11.6 ± 0.4 | 8.4 ± 0.2 |
| ECFP4 | Mean baseline | scaffold | -0.02 ± 0.00 | 12.2 ± 0.0 | 8.9 ± 0.0 |
| RDKit descriptors | XGBoost | scaffold | 0.52 ± 0.02 | 8.4 ± 0.1 | 6.0 ± 0.1 |
| RDKit descriptors | Random forest | scaffold | 0.51 ± 0.01 | 8.5 ± 0.1 | 6.1 ± 0.1 |
| RDKit descriptors | Ridge | scaffold | 0.32 ± 0.02 | 10.0 ± 0.1 | 7.5 ± 0.1 |
| RDKit descriptors | Neural network (MLP) | scaffold | 0.24 ± 0.09 | 10.6 ± 0.6 | 7.8 ± 0.4 |
| RDKit descriptors | Mean baseline | scaffold | -0.02 ± 0.00 | 12.2 ± 0.0 | 8.9 ± 0.0 |
