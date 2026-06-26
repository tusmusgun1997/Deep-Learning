# Hyperparameter Tuning — Finding the Best Network Configuration

## Parameters vs. Hyperparameters

*   **Parameters:** Weights ($W$) and biases ($b$). They are learned **automatically** by the model during the training process using backpropagation.
*   **Hyperparameters:** Configurations set **manually** by the developer before training starts. They control the behavior of the learning algorithm and the structure of the model.
    *   *Examples:* Learning rate, number of hidden layers, number of neurons per layer, epochs, activation function, dropout rate, weight decay penalty, optimizer choice.

---

## 4 Main Tuning Strategies

### 1. Grid Search
Exhaustively tries every single combination of hyperparameters in a specified set.
*   **Pros:** Simple to implement, guarantees coverage of the defined grid.
*   **Cons:** Extremely slow and computationally expensive (curse of dimensionality). Trashing resources on combinations where one parameter ruins performance.

### 2. Random Search
Randomly samples hyperparameter values from defined probability distributions.
*   **Pros:** Statistically superior to Grid Search. It doesn't waste evaluations on unimportant hyperparameters and checks many more unique values.
*   **Cons:** Still completely blind; does not learn from past trials.

### 3. Bayesian Optimization
Builds a probabilistic surrogate model (like a Gaussian Process) to map hyperparameters to final validation performance. It uses an **acquisition function** to decide which configuration to evaluate next (balancing exploration and exploitation).
*   **Pros:** Extremely efficient; finds optimal configurations in very few trials.
*   **Cons:** Harder to implement from scratch.

### 4. Hyperband (Successive Halving)
Starts by training a large number of random configurations for a tiny number of epochs. Then, it discards the worst performing half and doubles the budget (epochs) for the remaining half, repeating this until only the best configurations remain.
*   **Pros:** Blazing fast, allocates compute resources to promising setups early.

---

## Practical Tuning Tips

1.  **Search Logarithmically:** For parameters like learning rate ($\alpha$) or weight decay, search on a log scale: `[0.1, 0.01, 0.001, 0.0001]`, rather than a linear scale.
2.  **Coarse-to-Fine Search:** Start with a wide search grid running for a small number of epochs. Once you find a promising region, zoom in with a narrower grid and run for more epochs.
3.  **Monitor Val Loss:** Never select hyperparameters based on training loss. Always evaluate on a separate **Validation Set** to avoid selecting a model that overfits.
