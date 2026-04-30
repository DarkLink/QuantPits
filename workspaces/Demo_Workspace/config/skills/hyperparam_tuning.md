# Hyperparameter Tuning Knowledge

## Underfitting
- Few epochs with early stop → increase `n_epochs`
- Low IC with stable trend → increase model capacity

## Overfitting
- Full epochs + low IC → increase `dropout`, reduce `n_epochs`
- IS/OOS gap → increase regularization
