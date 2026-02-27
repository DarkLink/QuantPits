# Comprehensive Analysis Module

This module exists to provide professional, multi-dimensional auditing perspectives for the quantitative system, encompassing but not limited to: individual model validity decay, marginal contribution discrepancies of fused ensembles, authentic execution slippage friction, and traditional risk evaluations mapped natively from actual capital trajectories.

The centralized entry access script operates as: `scripts/run_analysis.py`

---

## I. Overview of Module Architecture

The execution controller script spawns 4 mutually disconnected component analyzers (`scripts/analysis/`) synthesizing a fully compiled Markdown intelligence report:

1. **Single Model Performance (`single_model_analyzer.py`)**: Tracks raw predictive intensity metrics and degeneration velocity curves primarily relying upon Rank IC evaluations paired with temporally decaying T+1 through T+5 half-life projections. Sequentially reviews ICIR distributions, Decile Spread hierarchies, alongside long-only bounds isolated via exclusively Top asset subset parsing matrices (Long-only IC).
2. **Ensemble Dynamics & Correlation (`ensemble_analyzer.py`)**: Assesses spearman correlations tracing identical evaluation spans bounded cross-sectionally across logic variants. Explicitly applies empirical Leave-One-Out validation calculating strict **marginal Sharpe contributions** for each atomic submodule, concurrently assessing native metrics generated simulating basic equalized distributions bounding the holistic ensemble mechanism.
3. **Valid Micro Friction Tracing (`execution_analyzer.py`)**: Segments discrete, authenticated post-trade vectors explicitly deriving native **Delay Cost** estimates (e.g. native variance bounds resulting exclusively via closing bounds against corresponding opening gap constraints of the next trading day) parallel with mapped **Execution Slippage** estimates (deviances resulting structurally tracking gap sequences against finalized trade actions parameters). Assesses extreme temporal intraday excursions tracing MAE/MFE metric bounds explicitly.
4. **Portfolio Capital Constraints (Risk) Risk Evaluations (`portfolio_analyzer.py`)**: Aggregates trailing positional valuations mapped uniquely using native system arrays combining chronological closing net valuation values merged transparently alongside sequential transactional capital distributions bounding the fundamental `trade_log_full.csv` datasets. Computes explicitly sanitized compounding elements yielding CAGR, Absolute Returns, Risk-free (1.35%) corrected Sharpe derivations, Volatility indices, robust Drawdown scaling, discrete capitalization efficiencies bounded by Turnover constraints and Win/Loss Ratios alongside systematic cross-sectional Barra Style Exposures regression attributes.

---

## II. Primary Usage Directives

Standard utilization environments typically trigger explicitly when structural array mutations alter targeting behaviors or immediately sequential to temporal database expansions adding chronological depth traces.

### 1. Routine Exhaustive Diagnostics

```bash
cd QuantPits

# Triggers report serialization assessing comprehensive cross-logic frameworks mapped internally against stated algorithms explicitly
python quantpits/scripts/run_analysis.py \
  --models gru_Alpha158 transformer_Alpha360 TabNet_Alpha158 sfm_Alpha360 \
  --output output/analysis_report.md
```

### 2. Parameters Breakdown

| Argument Switch | Requirement Designation | Target Logic Mapping Context |
|------|-----------|------|
| `--models` | Mandatory | Discrete mapping targets instructing assessment boundary generation explicit algorithm subsets e.g. `gru_Alpha158`. |
| `--start-date` | Optional | Temporal beginning parameter bounds (YYYY-MM-DD format). Safely inferred bounding dynamically native persistence arrays by default explicitly. |
| `--end-date`| Optional | Temporal ceiling parameter bounds (YYYY-MM-DD format). Bounds inherited autonomously mapping outputs absent targets natively. |
| `--market`| Optional | Target asset scaling universe mapping targets implicitly (defaults explicitly targeted `csi300`). |
| `--output` | Optional | Forces IO persistence output routines generating exact discrete markdown file strings. Bypasses terminal exclusively printing sequences typically native defaulting absent paths. |

### 3. Interactive Visual Dashboard 

Alongside serialized markdown reports, the system delivers multi-dimensional visual interactions leveraging fully unified interactive arrays rendered natively through Streamlit + Plotly elements. This visual architecture seamlessly queries identical data pathways immediately extracting explicit insights highlighting execution metrics alongside multi-dimensional performance attributes explicitly tracking factor decay and style drifts seamlessly.

```bash
cd QuantPits

# Spawns visualization matrices explicitly (Accessed natively by bridging browsers evaluating http://localhost:8501 bounds)
streamlit run dashboard.py
```

Operating internally bounding visual representations, developers natively govern temporal chronological windows (Start Date, End Date) alongside natively mapped Benchmark index parameters. Processing executions output explicit multi-segment architectures natively:
- **Macro Performance & Risk (Macro)**: Sequential logarithmic tracing returns bounded by underwater interactive maps mapping natively 20-day Sharpe and Alpha chronological progression indices explicitly.
- **Micro-Execution & Friction**: Scatter density evaluation matrices mapping combined MAE/MFE parameters explicitly targeting discrete chronological trade sets interactively tracking granular deviances highlighting execution anomalies and slip constraints interactively mapping sequential histogram density overlays against substitution bounds analysis explicitly.
- **Factor Exposure & Attribution**: Explicit Barra factor drift distributions evaluated utilizing dynamic rolling parameters (20-day regression matrices) distinctly mapped producing visual multi-level component representations splitting explicit Beta dependencies alongside stylistic parameters and distinct granular alpha deviances explicitly.
- **Holdings & Trade Analytics**: Concentration monitoring checks bounding singular position distributions mapping natively alongside FIFO-adjusted trade mapping logic exposing explicit granular profitability arrays structured structurally evaluating mapped day-vs-month distributions representing explicit probabilistic heat grids natively mapping empirical strategy success matrix layouts explicitly.

### 4. Rolling Analysis Dashboard

To counter the inherently obfuscated "style drift" and "gradual performance decay" vectors invisible bounding discrete end-to-end metrics exclusively, the architecture provides designated sequential rolling environments natively mapping explicit chronological intervals scaling utilizing moving window architectures mapping exclusively sequential single step environments.

1. Initially execute generation sequences mapping requested target time window parameter matrices (example matrices isolating 20 and 60 parameters distinctly):
```bash
cd QuantPits
python quantpits/scripts/run_rolling_analysis.py --windows 20 60
```

2. Follow directly executing explicit visualization rendering systems mapping targeting sequential ports uniquely:
```bash
streamlit run rolling_dashboard.py --server.port 8503
```

- Target monitoring scopes explicitly encompass:
  1.  **Rolling Factor Exposure**: Dynamic environments mapping stylistic scaling tracking Size matrices alongside explicit Momentum / Volatility constraints targeting deviation or concentrated parameter densities explicitly mapping.
  2.  **Rolling Return Attribution**: Stacked evaluation rendering bounding discrete daily components dividing Beta mapping against authentic Style Alpha extraction arrays mapping isolated authentic unassociated Idiosyncratic Alpha variables strictly.

### 5. Automated Rolling Health Report

Extrapolating bounds optimizingTo further reduce the time spent manually monitoring the dashboard, the system now includes a one-click "Rolling Health Check" diagnostic tool. This script intelligently extracts and compares multi-dimensional metrics across different rolling windows (`output/rolling_metrics_{20/60}.csv`), detects anomalies, and automatically generates a Markdown summary with recommended actions.

```bash
cd QuantPits

# Notice: execution explicitly demands active sequential parameters strictly rendered natively utilizing data extraction loops chronologically mapping ahead explicitly.
python quantpits/scripts/run_rolling_analysis.py --windows 20 60
python quantpits/scripts/run_rolling_health_report.py
```
This routine triggers bounding output mapped specifically generating `output/rolling_health_report.md` structures explicitly evaluating 3-stage multi-criteria alerts native architectures dynamically mapping explicit boundary exceptions explicitly tracking:
1. **Z-Score Exclusion Boundaries (Friction limits)**: Real-time checks identifying explicitly whether immediate operational parameters suffer explicit gap constraint deviances exceeding negative parameters trailing past average 60-day scopes bounding variance distributions explicitly triggering structured system alerts natively identifying structural execution failures dynamically.
2. **Moving Average Deviances (Alpha Decay)**: Calculates utilizing 5-day explicit short intervals identifying immediate Idio Alpha collapses beneath 60-day Idio matrices explicitly mapping native algorithm degradation identifiers indicating total baseline component failure scaling trajectories directly mapped explicitly.
3. **Threshold Boundary Ruptures (Factor Drift)**: Diagnostic routines evaluating discrete stylistic matrices like exact Size representations bounding against explicit trailing historical representations mapping exact highest bounds evaluating explicit constraint triggers identifying native market structure anomalies dynamically indicating explicit neutral mapping extraction imperatives dynamically.

---

## III. Diagnostic Interpretation & Corrective Execution Constraints

While inspecting generated `analysis_report.md` metrics, actively supervise explicit structural threshold warning matrix indicators evaluating systemic actions mapping natively across domains tracking explicitly mapping:

### 1. Core Model Efficacy (Model Quality)
- **Rank IC Mean & ICIR**:
  - **Nominal bounds**: Persistent mapping where IC > 0.035 explicitly mapping combined against bounds tracing ICIR > 0.15 identifies robust Alpha matrices generating valid outputs continuously.
  - **Correction Triggers**: Metrics explicitly generating `T+1 IC` trailing heavily indicating nominal collapses identifying explicit <0.01 parameters mapping strictly over persistent scopes require immediate exclusionary directives mapped ceasing internal dependencies immediately invoking explicit `incremental_train.py` target overrides explicitly mappings.
- **Decile Spread**: Identifies expected yield differentiation natively capturing Top 10% predictions evaluated inversely evaluating Bottom 10% bounds natively (measured evaluating sequential bounds natively mapping T+1 environments). Authentic robust matrices continuously highlight definitively mapped positive yields natively.
- **Long-Only IC (Top 22)**: Confines explicit validation mapping predictions utilizing strict active subsets structurally exclusively representing absolute execution parameters bounding Top instances explicitly against absolute resultant yields utilizing strict Spearman regression representations. Fractional negative outputs routinely identified bounding extreme Top segments exclusively representing extreme distribution structures identifying strictly extreme biased false preferences structurally demanding deep scrutiny bounding models mapping native negative dependencies strictly explicitly.
- **IC Decay**: Robust prediction parameters ideally decay uniformly scaling backwards mapping originating bounding extreme density peaks identifying exclusively explicit boundaries natively mapped evaluating purely explicit mapping trajectories evaluating smooth regression slopes scaling natively.

### 2. Correlation Mapping & Isolation Vectors (Ensemble Correlation)
- **Target Constraint Profile**: Spearman matrices optimally mapping natively across scopes evaluating parameters strictly identifying `0.2` mapping strictly `0.5` bounding ranges identifying highly distinct unassociated parameter logic explicitly evaluating. Bound exceptions >0.5 indicate strict identical representations indicating mapping saturation uniquely.
- **Procedural Application**: Target bounds evaluating exact output mapping `Marginal Contribution to Sharpe` exclusively identifying outputs evaluating strictly identifying instances evaluating "Drop `A` -> impact on Sharpe: `+0.2`", indicating explicit exclusionary dependencies effectively driving explicit total combination outcomes upwards explicitly validating noise suppression natively removing matrix A targeting exactly Equal parameters specifically or manually editing parameter scope exactly isolating algorithm explicitly mapping actively tracking.

### 3. Execution Dependency Constraints (Execution Friction & Path Dependency)
- **Cumulative Trace Elements (Total Friction)**: Valid robust total operational friction evaluating exact values evaluating parameters tracking bounds exclusively mapping <0.2% identify exceptional execution trajectories natively scaled.
  - **Evaluation Thresholds**: Instances identifying recurrent **Execution Slippage variables displaying gross negative ratios** (exceeding >1% limits consistently) identifies exclusively erratic extreme limits evaluating executing orders exclusively acquiring asset boundaries tracking maximum variance natively evaluating exactly native bounds explicitly demanding execution protocol constraint moderation executing strictly parameters uniquely excluding artificial ex-rights distortions generating anomalies mapping data vectors correctly mapping evaluating metrics exclusively natively mapping bounds strictly mapping precisely.
- **Absolute Value Allocations (Absolute Slippage Amount)**: Converts ratio variables generating numeric absolute RMB scaling bounds identifying execution overhead definitively tracking actual cash value loss mapping natively strictly isolating dependencies exclusively assessing single day order bounds exactly finding origin constraints exclusively exactly identifying natively explicitly generating parameters.
- **Market Liquidity Bounds (ADV Participation Rate)**: Native system logic actively monitors exact execution quantities converting parameters scaling exclusively against holistic asset native absolute daily volume transactions exactly tracking unadjusted RMB equivalent logic mapping natively exactly identifying market participation ceilings explicitly mapping exactly 0.5% averages generating negligible resistance uniquely bounding extremes natively monitoring bounding 15%-25% mapping directly indicating terminal execution market threshold ceilings definitively indicating immediate scaling limit requirements fundamentally directly bounding.
- **Explicit Friction Calculations**: Explicit evaluation mappings strictly identifying specific commission, taxation parameters dynamically injecting bounds bounding real yield elements identifying bounds extracting explicit dividend distribution variables natively evaluated creating genuine net bounds explicitly mapping explicitly natively tracking explicitly generating constraints actively generating results evaluating precisely directly extracting net impact bounds precisely tracking distinctly.
- **Execution Trajectory Deviations (Order Discrepancy & Substitution Bias)**:
  - **Substitution Bias Loss**: Bounds explicitly tracing simulated opportunity vectors matching 5-day predictive evaluation traces exclusively comparing original primary candidates natively unfulfilled evaluating explicitly bound tracking alternate acquired parameters structurally yielding absolute variance components measuring explicit execution loss generated naturally identifying boundary anomalies explicitly matching target validation requirements definitively evaluating highly constrained limit assets natively avoiding executing non-tradable assets distinctly highlighting boundary evaluation tracking distinctly metrics actively isolating tracking evaluating explicitly validating parameters evaluating natively actively generating exact mapping precisely distinguishing results generating explicitly distinctly tracking.

### 4. Holistic Capital Tracking (Return, Risk & Efficiency)

Metrics uniquely evaluated utilizing absolute validated transactional ledgers (`daily_amount_log_full.csv` alongside `trade_log_full.csv`) extracting exactly native execution constraints matching absolute cashflow removal yielding exactly precise compounding evaluation indices natively mapped: 

- **Primary Parameters (CAGR & Max Drawdown)**: Uniquely bound utilizing absolute metrics natively aligning directly corresponding execution tracking systems natively extracting values explicitly mapping bounds uniquely tracking exact vectors mapping perfectly explicitly tracking exactly if extreme deviations structurally map verify recent unparsed operational deposit vectors exclusively avoiding incorrect output bounds evaluating purely. *Sharpe constraints intrinsically apply native 1.35% (Annual) risk-free boundaries explicitly mapped evaluating strictly utilizing precise native matrices.*
- **Systematic Deviation Tracking (Relative Risk to Benchmark)**:
  - **Tracking Error (TE)**: Bounding metric explicitly generating target constraints evaluating absolute structural variance metrics distinctly identifying deviations distinctly tracing standard tracking CSI300 targets generating exact active bounds evaluating exclusively lower variables indicate tighter explicit tracking correlations mapping baseline tracking dependencies definitively evaluating exactly generating boundaries exclusively avoiding extreme risk configurations structurally isolating bounds tracking correctly.
  - **Information Ratio (IR)**: Exactly `Active Annual Outperformance / Annualized Tracking Error` isolating structurally generating absolute metrics tracing precise deviation compensation yields dynamically indicating exact structural parameters creating unique limits creating precise constraint values strictly.

- **Capital Velocity Metrics (Efficiency)**:
  - **Turnover_Rate_Annual (Yearly Reallocation Velocity)**: Native day-frequency algorithms routinely operate evaluating bounds inherently targeting 1000% - 2500% matrix bounds evaluating structural ranges mapped strictly extracting exact daily variables targeting discrete limits calculating boundaries identifying highly unstable environments demanding execution throttle configurations specifically verifying executing environments yielding alpha elimination parameters uniquely extracting evaluating limits specifically mapping identifying thresholds specifically measuring values effectively eliminating over-trading natively explicitly preventing excessive boundary mapping dynamically verifying bounds generating bounds identifying exact matrices explicitly preventing excessive trading frequencies actively mapping dynamically tracking natively limits creating precisely values strictly identifying precise results finding boundary limits exactly locating explicitly measuring preventing explicit decay accurately.

### 5. Trade Classification & Manual Impact

This metric isolates and cleanly delineates actual trading logs (`trade_log`) cross-referenced against original system-generated operational rank lists (`buy/sell_suggestion`). By strictly tracking execution provenance, the analyzer prevents human intervention noise from distorting quantitative system attribution evaluations natively:

- **Classification Distribution**:
  - **SIGNAL (S)**: Direct model-initiated signals natively tracing executing assets present within recommendation ranking limitations strictly matched against the physical executed quantity limits dynamically. This validates absolute rigorous quantitative constraints strictly mapping evaluating.
  - **SUBSTITUTE (A)**: Alternative structural targets tracing matched recommendation vectors mapped uniquely situated below daily absolute ranking thresholds executing exclusively mapped compensating targets evaluating unfulfilled execution boundaries structurally mapping implicitly executing compromises dynamically mapping strictly evaluating structural substitution loss indices dynamically extracting explicitly tracking implicitly tracking effectively native limits tracking liquidity bottlenecks dynamically isolating mapping limitations exclusively precisely extracting matching specifically.
  - **MANUAL (M)**: Entirely manually intervened logic entirely outside generated recommendation constraints natively extracting vectors implicitly evaluating purely discretionary actions exactly identifying isolation limits mapped avoiding pollution variables natively explicitly. **All execution friction (e.g. slippage) calculations explicitly remove this parameter segment from absolute quantitative evaluations purely validating strategy isolation vectors implicitly analyzing parameters strictly filtering natively tracking explicitly avoiding distortions isolating natively explicitly mapped.**
  
- **Quantitative-Only Performance**:
  - Excludes strictly separated MANUAL trade instances targeting strictly isolated signal performance elements natively generating "What-If" algorithm isolated trajectories identifying authentic performance logic natively generating uncontaminated strategy valuations dynamically bounding metrics effectively identifying strictly isolated algorithmic parameters dynamically validating absolute isolated capability explicitly accurately defining precise limits strictly tracking natively avoiding arbitrary pollution tracking defining precise parameters explicitly extracting logic isolating performance generating results actively isolating distinctly limiting accurately identifying bounds distinctly.
  
- **Manual Trade Details**:
  - Specifically isolating transactional ledgers exposing discrete manual executions bounding exactly dates, directions, and transactional arrays implicitly isolating tracking bounding evaluating strictly analyzing subsequent performance tracking limits actively generating metrics exclusively mapping native human interference tracking results generating exactly empirical behavioral analytics natively identifying exact parameters mapping limits distinctly tracing precisely identifying accurately executing results dynamically measuring constraints mapping precisely evaluating implicitly finding actual limits identifying distinctly tracking results extracting exact parameters natively explicitly capturing explicit tracking.
