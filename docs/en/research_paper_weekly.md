# Independent weekly paper account

`python -m quantpits.scripts.paper_trade` supports one-time bootstrap, planning, open settlement, close valuation, explicit events and reports. Use one workspace per account. Accounts may read shared signals and market data from another workspace. Paired roles, seals and capsules are not prerequisites. Existing research B0/B1 quantity and fee semantics remain unchanged.

## Configuration and sources

The account uses `market` from its `config/model_config.json` and `topk_dropout` parameters from `config/strategy_config.yaml`: `topk`, `n_drop`, `buy_suggestion_factor`, and optional `sell_out_of_universe`. Membership comes from the provider's `instruments/<market>.txt` effective intervals at the selected anchor. Holdings do not define the universe.

Create `config/paper_config.json` in the account workspace. All values below are synthetic examples, including fee rates. Supply account-specific fees, rather than backtest fees. Normal-limit declarations must apply to the selected execution session. Supply explicit references for special rules or corporate actions; unknown rules cause an individual skip.

```json
{
  "account_id": "example_account",
  "signal_source": {
    "workspace": "/path/to/signal_workspace",
    "mlflow_uri": "sqlite:////path/to/signal_workspace/mlflow.db",
    "experiment_id": "<EXPERIMENT_ID>",
    "recorder_id": "<RECORDER_ID>",
    "lineage_files": ["data/training_sources/chosen_fit.json"]
  },
  "execution": {
    "base_fee": "4.00",
    "base_rate": "0.00008",
    "sell_tax_rate": "0.0007",
    "transfer_rates": {"SH": "0.00002", "SZ": "0"},
    "lot_size": 100,
    "cash_reserve": "0.00",
    "slippage_rate": "0",
    "observed_max_gross": "15000.00"
  },
  "market_rules": {
    "source": "explicit rules checked for the selected session",
    "boards": {
      "SH_MAIN": {"normal_limit": true, "limit_rate": "0.10"},
      "SZ_MAIN": {"normal_limit": true, "limit_rate": "0.10"},
      "CHINEXT": {"normal_limit": true, "limit_rate": "0.20"}
    },
    "instruments": {}
  }
}
```

The base fee is `max(base_fee, gross × base_rate)`. Do not add regulatory fees already included in that base. Sell tax and separately charged exchange transfer fees are rounded individually to cents, then summed. Set `base_rate=0` explicitly for a flat base. Orders above `observed_max_gross` carry an estimate flag. These example tax rates do not represent current market tax rates. This workflow supports ordinary 100-share buys and complete odd-lot exits; STAR Market, Beijing Exchange, ETFs, financing and nonzero extra slippage are outside its current scope.

A source can instead provide `"prediction": "output/predictions/chosen_signal.csv"`. CSV requires `datetime,instrument,score`; pickle requires a DataFrame/Series with those two index levels. Only the selected anchor is used. Missing scores remain visible in coverage and are not filled with zero. MLflow sources require an explicit experiment, a FINISHED recorder and local artifacts. SQLite is opened read-only, and file metadata is read directly. Neither backend is initialized or migrated. Tracking metadata and artifacts must belong to the selected signal workspace.

Every plan freezes source URI/recorder or file path/hash, actual universe, price evidence, and optional fit/reload/manifest contents. Later source changes do not change an existing plan. Override period inputs using `--signal-workspace`, `--signal-mlflow-uri`, `--recorder-id` with `--experiment-id`, or `--prediction`. Repeat `--lineage-file` for additional frozen source material. Shared model sources do not require retraining for each account.

## First and following periods

These are example dates; select dates explicitly for your own run. Fill in the account configuration above and its market/strategy files. The read-only starting snapshot uses prod_config format: matching `current_date` or `last_processed_date`, `current_cash`, and `current_holding`, where `value` is shares and `amount` is total book cost. An independently specified starting account may use the same format.

```bash
PAPER_WS=/path/to/paper_workspace
PROVIDER=/path/to/qlib_provider
SOURCE_ACCOUNT=/path/to/read_only_account/config/prod_config.json
ANCHOR=2024-05-10
TRADE_DATE=2024-05-13
VALUE_DATE=2024-05-17

python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  bootstrap --source-config "$SOURCE_ACCOUNT" --date "$ANCHOR"
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  plan --cycle week_one --anchor "$ANCHOR" --trade-date "$TRADE_DATE"

# Once execution-day data is available. Before that: WAITING_FOR_DATA, exit 2, no balance change.
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  settle --cycle week_one
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  value --date "$VALUE_DATE"
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  report --date "$VALUE_DATE" --since "$ANCHOR" --reference-config "$SOURCE_ACCOUNT" --save

# After settling the previous period and preparing a specific new signal with existing commands.
NEXT_ANCHOR=2024-05-17
NEXT_TRADE_DATE=2024-05-20
NEXT_SIGNAL=/path/to/signal_workspace/output/predictions/next_signal.csv
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  plan --cycle week_two --anchor "$NEXT_ANCHOR" --trade-date "$NEXT_TRADE_DATE" \
  --prediction "$NEXT_SIGNAL"
```

Bootstrap copies once, preserving costs while setting initial NAV from cash closes. Returns begin at that NAV. Training placeholder balances do not become paper balances. Subsequent periods continue the paper account's own cash/shares/cost. This CLI does not train models, generate model predictions or invoke the production pipeline. Prepare new signals with the existing training/ensemble commands, then consume them read-only.

Execution must follow the anchor. Known nontrading dates fail explicitly. Dates beyond the future calendar can be stored as `PENDING_CALENDAR`, then checked against the actual calendar before settlement. Missing whole-session data, or a fully missing requested open surface without explicit facts that all instruments are suspended, leaves settlement pending. No automatic date rolling occurs. Retire an unusable pending plan explicitly, and plan with a new cycle ID:

```bash
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  cancel --cycle week_two --reason "explicit change of selected execution date"
# Then plan week_two_revised with an explicitly selected trade-date.
```

Cancellation retains the original plan/reason and cannot reverse settled trades. An account has at most one pending cycle. Reusing cycle/event IDs with different facts fails explicitly.

## Execution and valuation

Planning reuses production position analysis, saving all `buy_candidates` before zero-quantity suggestion filtering. Suggested quantities are anchor-close budget estimates. Execution uses the selected cash open (Qlib open/factor, cent precision), sells before buys, and uses neither H nor L. Execution-day close is used only for valuation. There is no queue, volume limit or intraday partial-fill model.

Missing opens, known suspension, limits or unavailable shares leave sells unfilled while other orders continue. Unfilled sells generate neither cash nor slots. Same-day buys cannot be resold. Available cash consists of starting cash, actual simulated net sell proceeds and explicitly booked events. Slots are the lesser of planned buys and actual post-sale `topk` vacancies.

Each slot receives a fixed `(clean_cash - cash_reserve) / slots` budget rounded down to cents. Reserve must be nonnegative and no larger than clean cash. Buy the largest whole lot including fees within both this budget and remaining cash. Actual quantities may exceed or fall below suggestions. Missing/limited/unaffordable/out-of-universe candidates retain reasons and trigger ordered replacement without consuming slots. Exhausted slots/candidates leave cash; no outsiders or follow-up purchases are introduced.

An open at either price limit skips both sides as a conservative paper assumption. Reliable dated instrument limits take precedence. Otherwise, computation requires a supported category, an explicit normal-limit declaration, unchanged factor and a cash close from the preceding actual session. Configured percentages are rounded to the price tick. A code prefix identifies the supported category but does not establish normal-limit status. ST names do not determine percentages. Unknown rules, factor changes or missing references cause an individual skip.

Pass `--rules /path/to/session_rules.json` to plan or settle for dated rules. Use the same `source/boards/instruments` structure with an exact `date`. Instrument overrides support `suspended`, `normal_limit`, `limit_rate`, `reference_close`, or paired `limit_up/limit_down`. Supplements change trading constraints only; signal/order/predecessor selection remains frozen. Different rules cannot rerun a completed settlement.

`value --date` appends a separate close observation, without trades/fees; valuation works without rebalancing. A missing individual close may reuse a reliable saved valuation price and exposes `price_date/stale`. Without such a price, known market value is reported and NAV/returns are null. Missing whole-session data waits. Period returns use `--since`, defaulting to the previous observed date; specify the week baseline when adding daily observations. Two endpoints do not establish daily drawdown.

## Explicit events and actual reference

`event --file /path/to/event.json` registers one sourced event belonging to this account. Actual-account dividends are not copied automatically. This example is synthetic:

```json
{
  "event_id": "notice_example",
  "effective_date": "2024-05-15",
  "booking_date": "2024-05-15",
  "kind": "DIVIDEND",
  "cash": "12.30",
  "instrument": "SH600001",
  "quantity_delta": 0,
  "cost_delta": "0",
  "source": "confirmed entitlement and payment notice for this account"
}
```

Supported kinds are `DIVIDEND` (positive cash), `TAX` (negative cash), `EXTERNAL` (capital in/out), and `SHARES` (existing position quantity/cost change, zero cash). Events are booked by booking_date in the next plan/settle/value. Retrying an ID does not repeat income. Effective dates are retained. Late information requires an explicit booking_date at/after the account frontier and appends an adjustment; it does not rewrite history. Events cannot produce negative cash or invalid quantities. The operator confirms entitlement dates for this account. Unknown events remain disclosed as unmodeled.

Reports separate net external funding from `NAV change - net funding`; dividends/tax remain investment profit/loss. With external funding, NAV/initial NAV is not presented as unit return. The simple profit-excluding-net-funding ratio is not claimed as a time-weighted return.

`--reference-config` reads an actual snapshot only at the selected date. A date mismatch is `PENDING_UPDATE` and does not block paper. A current actual snapshot cannot establish interval cashflows. Supply `--reference-events` (an array in the same event format; `[]` explicitly confirms no known cashflows) and, for later periods, `--reference-start-config`. The first period uses the copied starting snapshot. Cash/shares/NAV still appear without these inputs, while actual interval return remains pending.

## Storage and recovery

The authoritative balance is derived from bootstrap and ordered records in `<workspace>/data/paper/account.json`. The journal retains source snapshot/hash, frozen plans, per-order fills/skips, events, cancellations and appended valuations. It does not write prod_config. `data/paper/account.lock` uses the existing execution lease scoped to this account. `report --save` creates derived `output/paper/report_<date>.json` and `.md` without changing the account.

New plans use current production order rules. Historical plans retain their saved candidate order, suggestions and sources. Readers check identity, dates, universe membership, quantity/amount structure, frozen prices and account constraints. Suggested amounts remain historical budget references, so production suggestion changes leave old plans available for settlement/valuation. Reading or reusing a saved plan preserves its contents and journal digest. Actual cash, fees, costs and predecessor chains remain checked by accounting replay.

Writes atomically replace the journal and reread its canonical path. Readers check source price evidence, account/predecessor/plan identity, order fees and after-state. Retry the same cycle after interruption: an uncommitted transition runs again, while a committed result is returned unchanged without another trade/fee. Do not recover by deleting the journal, clearing locks, replacing cash or bootstrapping again. A new account uses a new workspace without copying another account's pending plans/locks. Exit codes: success 0, waiting for data 2, input/contract error 1.
