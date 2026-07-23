# Strategic Bidding in Online Auctions

A data science and reinforcement learning project exploring how different bidding behaviours emerge in online auctions—and how an autonomous agent can learn to bid against them.

## Why this project matters

Online auctions combine incomplete information, strategic behaviour, and sequential decision-making. This project studies those dynamics from two complementary angles:

- **Empirical analysis** of a shill-bidding dataset to identify patterns in bidder behaviour.
- **Simulation and reinforcement learning** using a custom auction environment and a Deep Q-Network (DQN) agent.

The project sits at the intersection of **machine learning, behavioural modelling, game theory, and marketplace analytics**.

## What is included

- Exploratory analysis of auction and bidder-level features.
- Behavioural segmentation of bidders using unsupervised learning.
- A custom auction simulation with multiple bidder archetypes:
  - snipers,
  - incremental bidders,
  - jump bidders.
- A PyTorch DQN agent that learns bidding actions through experience replay and target-network updates.
- Training outputs and visualisations for reward, win rate, and profitability.

## Technical approach

### 1. Auction data analysis

The analysis uses features such as bidder tendency, bidding ratio, successive outbidding, early bidding, winning ratio, and auction duration to understand strategic bidding patterns.

### 2. Behavioural modelling

Clustering techniques are used to group bidders with similar behavioural characteristics. This provides a bridge between descriptive marketplace analytics and strategic choice modelling.

### 3. Reinforcement-learning environment

The custom environment represents each auction state using:

- time remaining,
- current price,
- the agent's private valuation.

The agent chooses among discrete bid increments and receives a terminal reward based on the profit earned when it wins.

### 4. Deep Q-Network

The DQN implementation includes:

- epsilon-greedy exploration,
- replay memory,
- a separate target network,
- discounted future rewards,
- neural-network optimisation in PyTorch.

## Repository contents

| File | Description |
|---|---|
| `Shill Bidding Dataset (1).csv` | Auction-level dataset used for behavioural analysis |
| Notebook files | Exploratory analysis, clustering, simulation, and reinforcement-learning experiments |

## Tools and technologies

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `PyTorch` · `Matplotlib` · `Reinforcement Learning` · `Clustering`

## Skills demonstrated

- Translating an economic decision problem into a machine-learning workflow.
- Designing a simulation environment from first principles.
- Implementing and training a DQN agent.
- Working with behavioural and marketplace data.
- Communicating technical work through reproducible notebooks.

## Potential extensions

- Compare DQN performance with rule-based and truthful-bidding baselines.
- Add discrete-choice models for bidder strategy selection.
- Evaluate robustness across auction formats and bidder populations.
- Refactor the notebook code into reusable modules with automated tests.

## Author

Built by [sincerelyindra](https://github.com/sincerelyindra), with interests in machine learning, economics, decision science, and strategic modelling.
