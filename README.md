# Choice Modelling in Online Auctions

**Modeling Strategic Bidding Styles in Online Auctions: A Discrete Choice Approach**

**Indra Kumar and Thandava Sai** · Indian Institute of Science (IISc)

This research project studies how bidders adopt strategic styles in online auctions and how a learning agent responds to those styles in simulation. It combines behavioral feature engineering, discrete choice modeling, exploratory segmentation, and deep reinforcement learning.

The empirical analysis uses the Shill Bidding Dataset to construct three strategy labels: **incrementalist**, **jump bidder**, and **sniper**. A separate Deep Q-Network (DQN) learns a bidding policy against scripted opponents. The accompanying report connects these analyses with Level-k reasoning from behavioral game theory.

## Research questions

- Can aggregate bidding indicators distinguish recognizable strategic styles?
- How do timing, bidding intensity, win orientation, and experience relate to strategy choice?
- Can a reinforcement learning agent learn effective responses to different opponent styles?

## Data and strategy definitions

The bundled [Shill Bidding Dataset](Shill%20Bidding%20Dataset%20%281%29.csv) contains **6,321 records**, **807 auctions**, **1,054 anonymized bidder IDs**, and **13 columns**. Auction durations are 1, 3, 5, 7, or 10 days. These counts describe the committed CSV.

The dataset originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/562/shill+bidding+dataset). Its original `Class` field distinguishes normal and shill bidding; the three strategy labels used here are constructed by the project.

| Strategy | Behavioral interpretation |
| --- | --- |
| Incrementalist | Repeated, relatively small bid increases |
| Jump bidder | Larger, discrete increases intended to change the auction's trajectory |
| Sniper | Bidding close to the auction deadline |

These are behavioral archetypes. The available data contain aggregate indicators rather than complete bid histories, so the empirical labels are proxies for these styles.

## Methodology

### 1. Behavioral feature engineering

The [choice modeling notebook](cm-project-1.ipynb) constructs six covariates:

| Feature | Construction in the notebook |
| --- | --- |
| `time_from_start_sec` | `Early_Bidding × Auction_Duration × 86,400` |
| `time_to_close_sec` | `(1 − Last_Bidding) × Auction_Duration × 86,400` |
| `burstiness_rate` | Reuses `Successive_Outbidding` |
| `bid_count_share` | Reuses `Bidding_Ratio` |
| `win_ratio_running` | Reuses `Winning_Ratio`; it is not recomputed as a historical running average |
| `exp_auctions` | Per-bidder cumulative row count after sorting by `Bidder_ID` and `Record_ID` |

Timing and experience are derived proxies; the dataset does not provide a verified chronological sequence of individual bids.

### 2. Strategy labeling and choice modeling

The labeling pipeline combines an attempted rule-based sniper seed with **K-means clustering (`K=3`)** on five behavioral features. Cluster centers are inspected and mapped to the three strategies, principally by time remaining until close.

Records are then expanded into a long-format choice table with one row per strategy alternative and a binary `chosen` indicator.

| Method | Implementation and purpose |
| --- | --- |
| Multinomial logit (MNL) | Regularized multinomial logistic regression in scikit-learn; predicts constructed strategy labels using the six covariates |
| Mixed logit | PyLogit model with a random sniper intercept, bidder-level mixing IDs, and 2,000 simulation draws |
| Gaussian mixture model (GMM) | Three-component feature segmentation; the notebook calls this a latent class model, but it does not estimate a latent-class choice likelihood |

### 3. Learning a bidding policy

The [DQN notebook](6-deep-reinforcement-learning-agent-for-strategic.ipynb) implements a separate synthetic auction environment. It does not train on the empirical strategy labels or fitted choice-model parameters.

| Component | Setting |
| --- | --- |
| Auction horizon | 100 discrete time steps |
| State | Normalized time remaining, current price, and the agent's private valuation |
| Actions | Wait, or increase the current price by 1, 5, or 10 |
| Reward | Private valuation minus final price when the agent wins; zero otherwise |
| Opponents | Scripted snipers, incrementalists, and jump bidders |
| Q-network | Two hidden layers of 64 ReLU units |
| Learning | Experience replay, target network, epsilon-greedy exploration, and Adam |
| Default experiment | 5,000 training episodes; 100 greedy evaluation episodes per setting |

## Reported results

### Strategy classification

The following metrics are preserved in the choice modeling notebook's saved outputs.

| Model | Training accuracy | Test accuracy | Test weighted F1 |
| --- | ---: | ---: | ---: |
| MNL | 0.993 | 0.995 | 0.995 |
| Mixed logit | 0.514 | 0.510 | 0.345 |

MNL uses a stratified 80/20 record split. Mixed logit uses a 70/30 split grouped by `choice_id`, keeping each record's alternatives together; this is not an auction-level or bidder-level holdout.

**Interpretation:** the MNL's 99.5% test accuracy measures agreement with labels constructed from the same behavioral features. It is evidence of label recoverability, rather than independent validation of real bidder intent. The models also use different splits, so these figures are not a controlled comparison.

### Reinforcement learning

The project report records the following results over 100 greedy evaluation episodes per setting. The committed DQN notebook contains the implementation without saved training or evaluation outputs.

| Evaluation setting | Win rate | Average reward |
| --- | ---: | ---: |
| Initial evaluation with one opponent of each type | 65% | 42.59 |
| Three snipers | 52% | 38.06 |
| Three incrementalists | 51% | 27.83 |
| Three jump bidders | 91% | 54.07 |
| Separate evaluation with one opponent of each type | 74% | 39.02 |

Rewards are in the simulator's valuation units. The two mixed-opponent rows represent separate evaluation samples. The report describes a learned policy combining small increments with occasional larger bids and varied timing; these findings apply to the specified simulated environment.

## Repository guide

| File | Contents |
| --- | --- |
| [Choice modeling notebook](cm-project-1.ipynb) | Feature engineering, strategy labeling, long-format construction, MNL, mixed logit, and GMM segmentation |
| [DQN notebook](6-deep-reinforcement-learning-agent-for-strategic.ipynb) | Auction environment, agent, training loop, evaluation, and visualization |
| [Dataset CSV](Shill%20Bidding%20Dataset%20%281%29.csv) | Input bidder–auction records |
| [LICENSE](LICENSE) | MIT license for the repository code |

## Getting started

### Create a local environment

The notebooks were authored on Kaggle. The example below targets **Python 3.11** and uses a scikit-learn version range that supports the notebook's existing `n_init="auto"` and `multi_class="multinomial"` arguments. The repository does not include a complete environment lockfile.

```bash
git clone https://github.com/sincerelyindra/Choice-Modelling.git
cd Choice-Modelling

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 statsmodels==0.14.4 \
  "scikit-learn>=1.2,<1.7" pylogit==1.0.1 matplotlib torch jupyterlab

jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

### Adapt the Kaggle paths

Before running `cm-project-1.ipynb` locally, replace **all** matching input and output paths:

| Existing path | Local replacement |
| --- | --- |
| `/kaggle/input/shill-bidding-dataset1/Shill Bidding Dataset.csv` | `Shill Bidding Dataset (1).csv` |
| `/kaggle/working/` | `./` |

The mixed-logit section also contains a cell that edits a hardcoded PyLogit installation path. For local execution, replace that cell with the following compatibility shim **before the first PyLogit import**:

```python
import collections
import collections.abc

collections.Iterable = collections.abc.Iterable
import pylogit
```

The notebook's `pip install pylogit` cell can be skipped when the package is already installed in the active environment.

### Run the notebooks

1. Run **`cm-project-1.ipynb`** in cell order. It generates `features_step3_fixed.csv`, `step4_strategy_labels.csv`, and `step5_long_format.csv`, then performs the model analyses.
2. Run **`6-deep-reinforcement-learning-agent-for-strategic.ipynb`** independently. Its `main()` function launches training, saves `auction_bidder_dqn.pth`, plots training and bidding behavior, and evaluates opponent compositions.

The DQN uses CUDA when available and otherwise runs on CPU. For an initial execution check, reduce the episode count in `main()`; use the original settings when investigating the reported experiment.

## Research assumptions and limitations

- **Label construction:** the sniper seed requires `time_to_close_sec <= 10` and `Auction_Bids == 1`. The bundled `Auction_Bids` field is normalized and never equals 1, so this rule seeds no records; the saved strategy labels come from clustering. Jump sizes are not directly observed.
- **Evaluation design:** clustering and MNL scaling occur before the train/test split. Further validation should fit preprocessing on training data and use bidder, auction, or chronological holdouts.
- **Mixed-logit identification:** the current specification gives repeated record-level covariates a shared coefficient across alternatives. These terms cancel from relative utilities; alternative-specific effects are needed to interpret behavioral slopes.
- **Exploratory segmentation:** the GMM uses all numeric columns of the long table, including `choice_id` and `chosen`. Feature selection and independent behavioral validation would strengthen interpretation of its segments.
- **Simulation scope:** opponents follow fixed rules, the agent can bid above its private valuation, and global random seeds are not set. Reported DQN results are stochastic simulation outcomes.

Useful extensions include complete bid traces, independently annotated strategies, revised choice specifications, repeated evaluations with uncertainty estimates, and adaptive opponents.

## Related study material

The accompanying report also discusses **Level-k reasoning**, a **Lowest Unique Positive Integer (LUPI) experiment involving 3,842 participants**, and a **Bayesian control-function extension**. These are supplementary parts of the study; the report PDF, raw LUPI responses, and Bayesian implementation are not committed to this repository.

The report provides these external notebook links:

- [Feature engineering and choice modeling — Kaggle](https://www.kaggle.com/code/indrakumar180122/cm-project-1)
- [Deep reinforcement learning agent — Kaggle](https://www.kaggle.com/code/indrakumar180122/notebookfdbc6716b3)
- [Bayesian control-function analysis — Google Colab](https://colab.research.google.com/drive/1ExSSGLIZLvNvXa8KAphZOQb6fbc36nqq?usp=sharing)

## License and data attribution

Repository code is released under the [MIT License](LICENSE). The dataset is distributed by UCI under **CC BY 4.0** and retains its separate attribution requirements.

Dataset citation: *Shill Bidding Dataset* (2020), UCI Machine Learning Repository. [doi:10.24432/C5Z611](https://doi.org/10.24432/C5Z611).
