# Hyperparameter Tuning Decision Knowledge

## Underfitting Signals (underfitting / severe_underfitting)

### Neural Network Models (LSTM, GRU, Transformer, TFT, LocalFormer, ALSTM, etc.)
- Training stopped early by early_stop → consider increasing `n_epochs` (no more than +50%)
- May also be caused by too-small early_stop patience → consider increasing `early_stop`
- Low absolute IC but stable trend → consider increasing `hidden_size` or `num_layers`
- Unstable training, loss oscillation → decrease `lr`

### Tree Models (CatBoost, LightGBM)
- Insufficient training iterations → increase `iterations` (no more than +50%)
- Low IC → increase `depth` or `num_leaves`

## Overfitting Signals (overfitting)

### Neural Network Models
- Full epochs trained + low IC → increase `dropout`, reduce `n_epochs`
- Training loss keeps decreasing but IC doesn't improve → increase `batch_size` (reduces gradient noise)

### Tree Models
- Large IS/OOS gap → increase `l2_leaf_reg`
- Full iterations but low IC → decrease `depth` or `num_leaves`

## IC Decay Signals (ic_decay)
- Check whether the model needs retraining (cross-reference stale_models signal)
- If recently retrained and still decaying → may be a hyperparameter issue rather than data staleness
- Consider adjusting learning rate or regularization

## Learning Rate Adjustment Principles
- Unstable training (loss oscillation) → decrease `lr`
- Convergence too slow → moderately increase `lr`, but no more than 2x current value
- Note the difference between `lr` and `learning_rate`: different model configs may use different field names

## Handling Global Default Values

When a parameter has the same value in almost all models (e.g., `early_stop=10` in 18/22 models), it is likely a system-level initial default rather than a per-model tuning result. In this case:

- **Do NOT batch-modify**: don't apply the same change to ≥3 models
- **Pick representatives**: select 2-3 worst-affected models (lowest IC or lowest epoch ratio) as an experiment group
- **Architecture matters**: NN models may benefit from increased `early_stop`, but Tree models are NOT affected by `early_stop` — never suggest `early_stop` changes for Tree models
- **Root cause first**: if a model stops at epoch 25 (early_stop=10, n_epochs=200), it means loss stopped improving around epoch ~15. Blindly increasing early_stop to 20 may be ineffective. Consider:
  - Is the learning rate too high, causing training instability?
  - Is model capacity insufficient (hidden_size / num_layers too small)?
  - Is the data too noisy (consider increasing batch_size)?
- **Experimental mindset**: adjustments to global defaults should be treated as experiments (confidence ≤ 0.65) that need Playground backtest validation
