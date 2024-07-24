This repository implements various online fair division algorithms to allocate items among agents while optimizing different fairness criteria. The data is sourced from Spliddit.

Algorithms:

Max Utility Algorithm: Allocates items to the agent with the highest value.
Least Satisfied Algorithm: Allocates items to the least satisfied agent.
Most Envy Algorithm: Minimizes the maximum envy among agents.
Nash Product Algorithm: Maximizes the Nash product of utilities.
Simulated Annealing: Approximates the global optimum using probabilistic techniques.
Optimal Envy-Freeness: Minimizes envy using linear programming.
Optimal Equitability: Minimizes the maximum utility difference using linear programming.
Criteria:

Utilitarian: Sum of utilities.
Egalitarian: Minimum utility among agents.
Equitability: Maximum utility difference.
Envy-Freeness: Maximum envy score.
Nash Product: Product of utilities.

The output of running the script is two matrices printed to the console: Average Results and Average Approximations. These matrices provide a summary of the performance of different algorithms across various fairness criteria.
