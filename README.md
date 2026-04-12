# Optimized_Fire_Sales
Based on a model of Feinstein and [Halaj (JEDC, 2023)](https://doi.org/10.1016/j.jedc.2023.104734), this code allows a user to run simulations of how price-mediated contagion arises when banks respond to funding shocks by selling securities to raise cash. They choose securities by solving an optimal portfolio program.

# Fire-Sale-Driven Financial Contagion Model (Feinstein–Halaj)

The python code repository is an implementation of a simulation framework for studying **fire-sale contagion and liquidity stress propagation** in an interbank system with overlapping asset portfolios. Banks respond strategically (or proportionally) to funding shocks by selling securities, causing price impacts that spill over to other banks.

The model draws on the framework developed by Grzegorz Halaj and Zach Feinstein. The interbank network generation follows [Halaj & Kok (2013), *Computational Management Science*](https://doi.org/10.1007/s10287-013-0168-4).

---

## Repository Structure

```
.
├── config.py                        # All user-facing parameters and toggles
├── FeinsteinHalaj_aux_functions.py  # Core mathematical functions and solvers
└── toy_model_FeinsteinHalaj.py      # Main simulation script
```

---

## File Descriptions

### `config.py`

Central configuration file. All key model parameters are defined here, so users can adjust the experiment without touching the simulation code.

| Parameter | Description |
|---|---|
| `NUMBER_OF_BANKS` | Number of banks in the simulated system (default: 20) |
| `NUMBER_OF_ASSETS` | Number of tradable asset types in portfolios (default: 40) |
| `SHOCKED_BANKS` | List of bank identifiers that receive the funding shock |
| `SHOCK_SIZE` | Percentage of unsecured funding that runs off (default: 20%) |
| `NSIM_Z` | How many versions of how to set z (external liabilities to be paid back) (default: 2) |
| `PRICE_IMPACT_PER_PRTC_SOLD` | Linear price impact in basis points per 1% of volume sold, which means by how many bps prices of a given security class changes if 1% of the total volume of these securities, summing up across all financial insititutions, is liquidated (default: 20 bps) |
| `PRICE_IMPACT_FUNCTION` | Asset-specific price impact overrides (dictionary: asset index → bps) |
| `NTATONNEMENT` | Number of tâtonnement iterations to reach equilibrium (default: 100) |
| `IFINTERNALISE` | 1: strategic, 0: banks decide about how and what to transact irrespective of the peers (default: 1) |
| `IFPROPORTIONALSELLING` | 1: proportional selling, 0: optimized using Feinstein and Halaj model (default: 0) |
| `RESPONSEMODE` | Bank response ordering: `0` = independent (i.e., in each round of the totannement, each bank transacts independently of other banks; in other words, they transact in parallel and only after the prices of the aggregated selling/ buying reacts), `1` = synchronous (i.e., banks in a preselected order, from 0 to $N-1$, transact and in each round of the tanonnement, each bank considers impact of transactions of all banks that already transacted in this round, `2` = random (similar to 1, but the order of transacting is randomised (permutation of $[0,...,N-1]$) |
| `IFCENTRALBANK` | Central bank intervention toggle: `0` = passive, i.e., absence of a regulator, `1` = active, i.e., the regulator purchasing securites sold by the banks to stabilize the prices |
| `CREATESTABLESYSTEM` | Generate and save a stable (pre-shock equilibrium) system (`1`) or use a freshly simulated one (`0`) |
| `IF_LOAD_FROM_PICKLE` | Load a previously saved stable system from disk (`1`) or simulate fresh (`0`) |
| `IFSTARTFROMSTEADY` | Start the shock experiment from the steady state (`1`) |
| `NRUNSTOSTABILIZE` | Number of pre-shock runs used to stabilize the system |
| `IF_DRAW_BIPARTITE` | Visualise the bank–asset bipartite network (`1`) |
| `THRESHOLD_BIPARTITE` | Minimum exposure threshold for drawing a bipartite edge |
| `IF_TURNOFF_ITER_STATS` | Suppress per-iteration statistics output (`1`) |

---

### `FeinsteinHalaj_aux_functions.py`

Library of auxiliary mathematical functions and solvers used by the main simulation script.

#### Data Utility Functions
- **`convertDictNumpyToPd`** / **`convertDictNumpyToPdWithKeysCols`** — Convert dictionaries of NumPy arrays into multi-indexed pandas DataFrames.
- **`convertDictNumpyToPd_sim`** / **`convertDictNumpyToPdWithKeysCols_sim`** — Same as above but indexed into a simulation sub-dictionary.

#### Portfolio Overlap Metrics and Generation of Portfolios approx. with a Given Similarity 
- **`funCosine`** — Computes the cosine similarity between two portfolio vectors.
- **`funPorfCosineSimilarity`** — Constructs a pairwise cosine similarity matrix across all banks' portfolios.
- **`funVecCosineSim`** — Generates a random non-negative vector with a prescribed cosine similarity to a reference vector.
- **`funOverlapPortfoliosV3`** — Generates overlapping bank portfolios by sampling from a uniform distribution and normalising, producing a bank × asset holdings matrix.

#### Interbank Network Generation
- **`mLendIBank`** — Simulates the interbank lending network following the Halaj & Kok (2013) algorithm. Uses a probability map  (e.g., a matrix capturing similarity in geographical presence of the banks) to probabilistically assign bilateral exposures between borrowers and lenders, subject to capacity constraints.

#### Price and Clearing Functions
- **`f`** / **`flin`** — Exponential and linear asset price impact functions. `flin` computes asset prices as a function of initial holdings `x` and current retained holdings `y`, with a linear price-impact coefficient `b`.
- **`FDAq`** — Fixed-point clearing algorithm (Eisenberg–Noe style) that finds the equilibrium payment vector and identifies defaulting banks given asset prices `q`, interbank liability matrix `Pi`, and promised payments `pBar`.
- **`FDAsale`** — Extended clearing algorithm that jointly solves for the equilibrium payment vector and the fire-sale liquidation quantities when banks sell proportional shares of their portfolios.

#### Optimisation
- **`best_response`** — Computes the strategic best-response portfolio `y` for each bank, solving a quadratic programme (via CVXPY / CLARABEL) that balances expected returns against price-impact costs and liquidity constraints.

#### Visualisation & I/O
- **`fun_bipirtite`** — Constructs a NetworkX bipartite graph linking banks to the assets they hold above a threshold.
- **`load_system_unpickled`** — Loads a pre-saved system of banks (i.e., their balance sheets, with risk and return parameters, interbank lending matrix, etc.) from a pickle file.

---

### `toy_model_FeinsteinHalaj.py`

The main simulation script. Reads parameters from `config.py`, calls functions from `FeinsteinHalaj_aux_functions.py`, and runs the full shock-and-response experiment.

#### Overview of the Simulation Pipeline

1. **Balance sheet generation** — Stylized bank balance sheets are either simulated from scratch (total assets drawn from uniform or Pareto distributions, securities portfolios generated with controlled overlap via `funOverlapPortfoliosV3`) or loaded from a pickle file.

2. **Interbank network** — The lending matrix `L` is constructed using `mLendIBank` with a uniform connection probability (`IBPROB`).

3. **System statistics** — Network centrality metrics (degree, betweenness, eigenvector centrality) and balance sheet summaries are computed and stored.

4. **Calibration (optional)** — If `IFTESTCALIB = 1`, the model calibrates bank-level risk-aversion parameters (`beta`, `gamma`) by solving a least-squares problem for each bank using CVXPY/CLARABEL, ensuring the pre-shock portfolio is a best response.

5. **Shock application** — A funding run-off shock of size `SHOCK_SIZE` is applied to the banks listed in `SHOCKED_BANKS`, increasing their payment obligations `pBar`.

6. **Tâtonnement iteration** — The system iterates up to `Ntatonn` steps. At each step:
   - Each bank solves a quadratic programme (`response_II`) to find its optimal retained portfolio `y`, subject to a liquidity constraint.
   - If `IFCENTRALBANK = 1`, the central bank solves its own quadratic programme (`response_CB`) to absorb excess supply and limit price dislocations.
   - Asset prices `q` are updated via the linear price impact function `flin`.
   - Interbank payments and defaults are resolved via `FDAq`.
   - The process repeats until convergence or the iteration limit is reached.

7. **Response modes** — Banks can update sequentially in a fixed order (`RESPONSEMODE = 1`), independently using last-round prices (`RESPONSEMODE = 0`), or in random order (`RESPONSEMODE = 2`).

8. **Selling strategies** — Banks can sell assets proportionally (`IFPROPORTIONALSELLING = 1`) or strategically via quadratic optimisation (`IFPROPORTIONALSELLING = 0`).

9. **Stable system creation** — If `CREATESTABLESYSTEM = 1`, the script runs `NRUNSTOSTABILIZE` rounds without a shock and saves the resulting equilibrium holdings of securities to disk as a pickle file, which can later be loaded as the starting point for experiments with funding shocks.

10. **Output collection** — Time series of prices (`q`), payment vectors (`p`), cash buffers (`c`), and bank/CB portfolio holdings (`y`) are collected across tâtonnement steps and (optionally) saved to disk.

---

## Dependencies

- Python 3.x
- `numpy`
- `pandas`
- `networkx`
- `cvxpy` (with the CLARABEL solver)
- `matplotlib`
- `scipy`
- `chardet`
- `pickle` (standard library)

Install Python dependencies with:
```bash
pip install numpy pandas networkx cvxpy matplotlib scipy chardet
```

---

## Usage

1. **Configure** the experiment in `config.py` (number of banks, shock size, shocked banks, price impact, etc.).
2. **Run** the main script:
   ```bash
   python toy_model_FeinsteinHalaj.py
   ```
3. To **reproduce experiments from a stable starting point**, first run with `CREATESTABLESYSTEM = 1` to generate and save the equilibrium system, then set `IF_LOAD_FROM_PICKLE = 1` for subsequent shock runs.

---

# Example of a setup
- Objective: create a banking system (semi-randomly), shock a subset of banks with a given shock to funding sources, analyse the optimal responses of banks raising cash to meet the obligations related to the funding shock and collect information about the price dynamics and changes in securities holding
- CREATESTABLESYSTEM:=1 -- force the code to create a stable, set IFPROPORTIONALSELLING:=0 (i.e., use Feinstein and Halaj model of how banks optimally transact to raise cash, instead of miopic proportional selling, like in Greenwood model) and IFINTERNALISE:=1 (to follow Feinstein and Halaj model of strategic interactions of banks)
- run the main script (pickled file "balance_sheet_stable_system"+period with the banking system is created)
- in config.py, set a list of banks SHOCKED_BANKS that are subject to funding shock, set the shock size SHOCK_SIZE, e.g., to 10 (means: 10%), increase the number of steps in tatonnement (NTATONNEMENT=200)
- force the code to load data from pickle (CREATESTABLESYSTEM:=0 and IF_LOAD_FROM_PICKLE:=1)
- run the main script: it will save output data in 5 files (assuming a default NSIM_Z=2), i.e.,
  1. q_ts_bank_BA_OPTM_strategic_{date}.xlsx: asset classes in columns, price dynamics in the steps of the tatonnement process in rows
  2. p_ts_bank_BA_OPTM_strategic_{date}.xlsx: banks in columns, interbank payments in equilibrium in tatonnement in rows
  3. x_collect_bank_BA_OPTM_strategic_{date}.xlsx: starting point holdings of securities (banks in rows, assets in columns)
  4. y_collect_sim0_bank_BA_OPTM_strategic_{date}.xlsx: holdings in equilibrium (after the last step of tatonnement) with shock structure 0
  5. y_collect_sim1_bank_BA_OPTM_strategic_{date}.xlsx: holdings in equilibrium (after the last step of tatonnement) with shock structure 1
 
- Remark: when setting NSIM_Z to larger number (e.g., 40) one can specify how the shock increases in each step (zstep*kkk in the inner loop of the toy_model code) and collect the results via dictionaries (e.g., q_mat_dict with prices or y_collect_dict with equilibrium holdings of securities)  

---

## Authors

- **Grzegorz Halaj** — [ECB author profile](https://www.ecb.europa.eu/pub/research/authors/profiles/grzegorz-halaj.en.html)

