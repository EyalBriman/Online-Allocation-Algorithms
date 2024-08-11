****Online Fair Division Algorithms****


This repository implements various online fair division algorithms to allocate items among agents while optimizing different fairness criteria. The data used for testing is sourced from Spliddit.

****Algorithms****

Max Utility Algorithm: Allocates each item to the agent who values it the most.
Least Satisfied Algorithm: Allocates each item to the agent with the lowest accumulated utility so far.
Future Most Envy Algorithm: Allocates each item to the agent who would experience the most envy if they did not receive it, considering future allocations.
Current Most Envy Algorithm: Allocates each item to the agent who currently experiences the most envy.
Utilitarian Least Satisfied Algorithm: Allocates each item to the agent who values it the most among the group of agents with the lowest accumulated utility.
Balance Cardinality Algorithm: Allocates each item to the agent with the fewest items in their bundle.
Nash Product Algorithm: Allocates items to maximize the product of utilities, prioritizing agents with zero utility to avoid multiplication by zero.

****Fairness Criteria****

Each allocation is evaluated based on the following fairness criteria, with optimal allocations calculated for comparison:

Utilitarian: The sum of utilities across all agents.
Egalitarian: The minimum utility among all agents.
Equitability: The maximum difference between the utilities of any two agents.
Envy-Freeness: The maximum envy score, calculated as the highest utility difference an agent feels towards another's allocation.
Nash Product: The product of all agents' utilities.
EF1 (Envy-Freeness up to One Item): The maximum envy an agent feels towards another's allocation after hypothetically removing one item from the envied agent's bundle.
Leximin: A weighted sum of normalized utilities, prioritizing lower utilities by scaling each utility with diminishing powers of 0.1.

****Output****

Running the script produces two matrices:

Average Results: A summary of how often each algorithm achieves the optimal score for each fairness criterion across all permutations of item allocations.
Average Approximations: The average difference between the algorithm's score and the optimal score for each fairness criterion.
These matrices provide a comprehensive evaluation of the performance of different algorithms under various fairness criteria.

