import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import permutations
import glob
from pathlib import Path
import random
from math import exp
gp.setParam('OutputFlag', 0)
from decimal import Decimal

def nash_product(allocation, utilities):
    product = Decimal(1)
    for i in range(len(utilities)):
        agent_utility = sum(utilities[i][k] for k in allocation[i]) if allocation[i] else 0
        product *= Decimal(int(agent_utility))
    return product

def extract_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    first_number = int(lines[0].split()[0]) 
    matrix_data = [line.split() for line in lines[2:2+first_number]] 

    matrix = np.array([[int(element) for element in row] for row in matrix_data])
    return matrix

def utilitarian(allocation, utilities):
    total_utility = 0
    for i in range(len(utilities)):
        agent_utility = sum(utilities[i][k] for k in allocation[i]) if allocation[i] else 0
        total_utility += agent_utility
    return total_utility

def equitability(allocation, utilities):
    utilities_list = [sum(utilities[i][k] for k in allocation[i]) if allocation[i] else 0 for i in range(len(utilities))]
    max_diff = max(utilities_list) - min(utilities_list)
    return max_diff 

def egalitarian(allocation, utilities):
    min_utility = float('inf')
    for i in range(len(utilities)):
        agent_utility = sum(utilities[i][k] for k in allocation[i]) if allocation[i] else 0
        min_utility = min(min_utility, agent_utility)
    return min_utility
    
def ef1(allocation, utilities):
    n = len(utilities)
    max_envy = float('-inf')
    
    for i in range(n):
        u_i_S_i = sum(utilities[i][k] for k in allocation[i])
        
        for j in range(n):
            if i != j:
                u_i_S_j = sum(utilities[i][k] for k in allocation[j])
                
                if len(allocation[j]) > 1:
                    u_i_S_j_min = float('inf')
                    for item in allocation[j]:
                        u_i_S_j_wo_item = u_i_S_j - utilities[i][item]
                        u_i_S_j_min = min(u_i_S_j_min, u_i_S_j_wo_item)
                else:
                    u_i_S_j_min = u_i_S_j
                
                envy = u_i_S_j_min - u_i_S_i
                max_envy = max(max_envy, envy)
    
    return max(0, max_envy)

def leximin(allocation, utilities):
    utilities_list = sorted([sum(utilities[i][k] for k in allocation[i]) for i in range(len(utilities))])
    max_utility = max(utilities_list) if max(utilities_list) != 0 else 1

    leximin_score = sum((utilities_list[i] / max_utility) * (0.1 ** i) for i in range(len(utilities_list)))
    
    return leximin_score

def envy_freeness(allocation, utilities):
    n = len(allocation)
    envy_scores = []
    
    for i in range(n):
        u_i_S_i = sum(utilities[i, item] for item in allocation[i])
        max_envy_i = float('-inf')
        for j in range(n):
            if i != j:
                u_i_S_j = sum(utilities[i, item] for item in allocation[j])
                envy_i_j = u_i_S_j - u_i_S_i
                max_envy_i = max(max_envy_i, envy_i_j)
        envy_scores.append(max_envy_i)
    
    return max(0,max(envy_scores))

def max_utility_algorithm_with_tiebreak(utilities, perm, agent_perm):
    allocation = [[] for _ in range(len(utilities))]
    for j in perm:
        max_val = max(utilities[i][j] for i in range(len(utilities)))
        candidates = [i for i in range(len(utilities)) if utilities[i][j] == max_val]
        chosen = min(candidates, key=lambda i: agent_perm.index(i))
        allocation[chosen].append(j)
    return allocation

def least_satisfied_algorithm_with_tiebreak(utilities, perm, agent_perm):
    allocation = [[] for _ in range(len(utilities))]
    current_utilities = [0] * len(utilities)
    for j in perm:
        agents_who_value = [i for i in range(len(utilities)) if utilities[i, j] > 0]
        if agents_who_value:
            min_val = min(current_utilities[i] for i in agents_who_value)
            candidates = [i for i in agents_who_value if current_utilities[i] == min_val]
        else:
            min_val = min(current_utilities)
            candidates = [i for i in range(len(utilities)) if current_utilities[i] == min_val]
        chosen = min(candidates, key=lambda i: agent_perm.index(i))
        allocation[chosen].append(j)
        current_utilities[chosen] += utilities[chosen][j]
    return allocation

def minimize_future_envy_algorithm_with_tiebreak(utilities, perm, agent_perm):
    n, m = utilities.shape
    allocation = [[] for _ in range(n)]
    for j in perm:
        envy_scores = []
        for i in range(n):
            current_envy = float('-inf')
            for k in range(n):
                if i != k:
                    u_i_S_i = sum(utilities[i][item] for item in allocation[i])
                    u_i_S_k = sum(utilities[i][item] for item in allocation[k])
                    envy_i_k = u_i_S_k + utilities[i][j] - u_i_S_i
                    current_envy = max(current_envy, envy_i_k)
            envy_scores.append(current_envy)
        max_val = max(envy_scores)
        candidates = [i for i in range(n) if envy_scores[i] == max_val]
        chosen = min(candidates, key=lambda i: agent_perm.index(i))
        allocation[chosen].append(j)
    return allocation

def minimize_current_envy_algorithm_with_tiebreak(utilities, perm, agent_perm):
    n, m = utilities.shape
    allocation = [[] for _ in range(n)]
    for j in perm:
        envy_scores = []
        for i in range(n):
            current_envy = 0
            for k in range(n):
                if i != k:
                    u_i_S_i = sum(utilities[i][item] for item in allocation[i])
                    u_i_S_k = sum(utilities[i][item] for item in allocation[k])
                    envy_i_k = u_i_S_k - u_i_S_i
                    current_envy = max(current_envy, envy_i_k)
            envy_scores.append(current_envy)
        max_val = max(envy_scores)
        candidates = [i for i in range(n) if envy_scores[i] == max_val]
        chosen = min(candidates, key=lambda i: agent_perm.index(i))
        allocation[chosen].append(j)
    return allocation

def utilitarian_least_satisfied_algorithm_with_tiebreak(utilities, perm, agent_perm):
    allocation = [[] for _ in range(len(utilities))]
    current_utilities = [0] * len(utilities)
    for j in perm:
        min_val = min(current_utilities)
        candidates = [i for i in range(len(utilities)) if current_utilities[i] == min_val]
        max_val = max(utilities[i][j] for i in candidates)
        best_candidates = [i for i in candidates if utilities[i][j] == max_val]
        chosen = min(best_candidates, key=lambda i: agent_perm.index(i))
        allocation[chosen].append(j)
        current_utilities[chosen] += utilities[chosen][j]
    return allocation

def balance_cardinality_algorithm_with_tiebreak(utilities, perm, agent_perm):
    allocation = [[] for _ in range(len(utilities))]
    for j in perm:
        min_len = min(len(allocation[i]) for i in range(len(utilities)))
        candidates = [i for i in range(len(utilities)) if len(allocation[i]) == min_len]
        chosen = min(candidates, key=lambda i: agent_perm.index(i))
        allocation[chosen].append(j)
    return allocation

def nash_product_algorithm_with_tiebreak(utilities, perm, agent_perm):
    n, m = utilities.shape
    allocation = [[] for _ in range(n)]
    allocated_items = set()
    for idx, j in enumerate(perm):
        if j not in allocated_items:
            allocation[agent_perm[idx % n]].append(j)
            allocated_items.add(j)
    remaining_items = list(set(range(m)) - allocated_items)
    for j in remaining_items:
        best_allocation = None
        best_product = float('-inf')
        for i in range(n):
            temp_allocation = [list(bundle) for bundle in allocation]
            temp_allocation[i].append(j)
            product = nash_product(temp_allocation, utilities)
            if product > best_product:
                best_product = product
                best_allocation = temp_allocation
        allocation = best_allocation
    return allocation

def optimal_utilitarian(utilities):
    model = gp.Model()
    n, m = utilities.shape
    x = model.addVars(n, m, vtype=GRB.BINARY)
    model.setObjective(gp.quicksum(utilities[i, j] * x[i, j] for i in range(n) for j in range(m)), GRB.MAXIMIZE)
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)
    model.optimize()
    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i].append(j)
    return allocation

def optimal_egalitarian(utilities):
    model = gp.Model()
    n, m = utilities.shape
    x = model.addVars(n, m, vtype=GRB.BINARY)
    min_util = model.addVar(vtype=GRB.CONTINUOUS)
    model.setObjective(min_util, GRB.MAXIMIZE)
    for i in range(n):
        model.addConstr(min_util <= gp.quicksum(utilities[i, j] * x[i, j] for j in range(m)))
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)
    model.optimize()
    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i].append(j)
    return allocation


def optimal_ef1(utilities):
    model = gp.Model()
    n, m = utilities.shape
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    max_envy = model.addVar(vtype=GRB.CONTINUOUS, name="max_envy")

    model.setObjective(max_envy, GRB.MINIMIZE)
    
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"item_{j}")

    for i in range(n):
        u_i_S_i = gp.quicksum(x[i, k] * utilities[i, k] for k in range(m))
        for j in range(n):
            if i != j:
                for removed_item in range(m):
                    u_i_S_j = gp.quicksum(x[j, k] * utilities[i, k] for k in range(m))
                    u_i_S_j_wo_item = u_i_S_j - utilities[i, removed_item] * x[j, removed_item]
                    model.addConstr(max_envy >= u_i_S_j_wo_item - u_i_S_i)

    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        print("The model did not converge. Here are the utilities:")
        return None
    
    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i].append(j)
    
    return allocation


def optimal_leximin(utilities):
    model = gp.Model()
    n, m = utilities.shape

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

    model.addConstrs(gp.quicksum(x[i, j] for i in range(n)) == 1 for j in range(m))
    model.addConstrs(u[i] == gp.quicksum(utilities[i, j] * x[i, j] for j in range(m)) for i in range(n))
    
    lexicographic_objective = gp.quicksum((0.1 ** i) * u[i] for i in range(n))
    
    model.setObjective(lexicographic_objective, GRB.MAXIMIZE)
    
    model.addConstrs(u[i] <= u[i + 1] for i in range(n - 1))
    
    model.Params.OutputFlag = 0  
    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        print("The model did not converge.")
        return None
    
    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i].append(j)
    
    return allocation
def optimal_envy_freeness(utilities):
    model = gp.Model()
    n, m = utilities.shape
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    max_diff = model.addVar(vtype=GRB.CONTINUOUS, name="max_diff")
    model.setObjective(max_diff, GRB.MINIMIZE)
    
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"item_{j}")
        
    for i in range(n):
        for j in range(n):
            if i != j:
                u_i_S_j = gp.quicksum(x[j, k] * utilities[i, k] for k in range(m))
                u_i_S_i = gp.quicksum(x[i, k] * utilities[i, k] for k in range(m))
                diff = u_i_S_j - u_i_S_i
                model.addConstr(max_diff >= diff, name=f"max_diff_{i}_{j}")
    
    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        print("The model did not converge. Here are the utilities:")
        print(utilities)
        return None
    
    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i].append(j)
    
    return allocation



def optimal_equitability(utilities):
    model = gp.Model()
    n, m = utilities.shape
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    max_diff = model.addVar(vtype=GRB.CONTINUOUS, name="max_diff")
    
    model.setObjective(max_diff, GRB.MINIMIZE)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(gp.quicksum(utilities[i, k] * x[i, k] for k in range(m)) \
                                - gp.quicksum(utilities[j, k] * x[j, k] for k in range(m)) <= max_diff)
                model.addConstr(gp.quicksum(utilities[j, k] * x[j, k] for k in range(m)) \
                                - gp.quicksum(utilities[i, k] * x[i, k] for k in range(m)) <= max_diff)
    
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"item_{j}_allocated")
    
    model.Params.OutputFlag = 0 
    
    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        return None
    
    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].x > 0.5:
                allocation[i].append(j)
    
    return allocation


def optimal_nash_product(utilities):
    model = gp.Model()
    n, m = utilities.shape

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
    log_u = model.addVars(n, vtype=GRB.CONTINUOUS, name="log_u")

    # Each item must be allocated to exactly one agent
    model.addConstrs(gp.quicksum(x[i, j] for i in range(n)) == 1 for j in range(m))

    # Each agent must get at least one item
    model.addConstrs(gp.quicksum(x[i, j] for j in range(m)) >= 1 for i in range(n))

    # Define utility for each agent
    model.addConstrs(
        u[i] == gp.quicksum(utilities[i, j] * x[i, j] for j in range(m)) for i in range(n)
    )

    # Add logarithmic constraints: log_u[i] = log(u[i])
    for i in range(n):
        model.addGenConstrLog(u[i], log_u[i], name=f"log_constraint_{i}")

    # Maximize the sum of log utilities (i.e., Nash Social Welfare)
    model.setObjective(gp.quicksum(log_u[i] for i in range(n)), GRB.MAXIMIZE)

    model.Params.OutputFlag = 0
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print("Model did not converge.")
        return None

    allocation = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i].append(j)

    return allocation


def generate_synthetic_utilities(n, m, alpha=0.5, total_tokens=1000):
    """
    Generate an (n x m) utility matrix with:
    - Each agent's utilities summing to `total_tokens`
    - Controlled correlation: alpha=0 (identical), alpha=1 (completely different)
    - Underlying distribution: Dirichlet-based
    """
    # Base profile (shared component) from Dirichlet, then scaled
    base_profile = np.random.dirichlet([1.0] * m)
    base_utility = (base_profile * total_tokens).round().astype(int)

    utilities = np.zeros((n, m), dtype=int)

    for i in range(n):
        # Generate a fully independent profile for this agent
        independent_profile = np.random.dirichlet([1.0] * m)
        independent_utility = (independent_profile * total_tokens).round().astype(int)

        # Mix the base and independent profiles
        mixed = alpha * independent_utility + (1 - alpha) * base_utility

        # Round and fix token sum (adjust for rounding error)
        rounded = np.round(mixed).astype(int)
        diff = total_tokens - np.sum(rounded)

        # Adjust by adding/subtracting from largest/smallest utility entries
        if diff != 0:
            sorted_indices = np.argsort(rounded)
            for idx in (reversed(sorted_indices) if diff < 0 else sorted_indices):
                adjustment = 1 if diff > 0 else -1
                if 0 <= rounded[idx] + adjustment <= total_tokens:
                    rounded[idx] += adjustment
                    diff -= adjustment
                    if diff == 0:
                        break

        utilities[i] = rounded+1

    return utilities

def optimal_for_criterion(criterion, utilities):
    if criterion == utilitarian:
        return optimal_utilitarian(utilities)
    elif criterion == egalitarian:
        return optimal_egalitarian(utilities)
    elif criterion == envy_freeness:
        return optimal_envy_freeness(utilities)
    elif criterion == equitability:
        return optimal_equitability(utilities)
    elif criterion == ef1:
        return optimal_ef1(utilities)
    elif criterion == leximin:
        return optimal_leximin(utilities)
    elif criterion == nash_product:
        return optimal_nash_product(utilities)
    else:
        raise ValueError("Unknown criterion")

algorithms = [
    max_utility_algorithm_with_tiebreak, 
    least_satisfied_algorithm_with_tiebreak, 
    minimize_future_envy_algorithm_with_tiebreak, 
    nash_product_algorithm_with_tiebreak, 
    minimize_current_envy_algorithm_with_tiebreak, 
    utilitarian_least_satisfied_algorithm_with_tiebreak,
    balance_cardinality_algorithm_with_tiebreak
]

criteria = [
    utilitarian, 
    egalitarian, 
    envy_freeness, 
    equitability, 
    nash_product, 
    ef1, 
    leximin
]


def process_instance_with_tiebreak(utilities):
    n, m = utilities.shape

    # All permutations of item orderings and agent orderings
    perms = list(permutations(range(m)))
    agent_perms = list(permutations(range(n)))

    min_percents = np.full((len(algorithms), len(criteria)), np.inf)
    max_percents = np.zeros((len(algorithms), len(criteria)))

    # Precompute optimal allocations and scores once per criterion
    optimal_allocations = {
        criterion: optimal_for_criterion(criterion, utilities)
        for criterion in criteria
    }
    optimal_scores = {
        criterion: criterion(optimal_allocations[criterion], utilities)
        for criterion in criteria
    }

    for agent_perm in agent_perms:
        tie_break_results = np.zeros((len(algorithms), len(criteria)))

        for perm in perms:
            for i, algorithm in enumerate(algorithms):
                allocation = algorithm(utilities, perm, agent_perm)
                for j, criterion in enumerate(criteria):
                    score = criterion(allocation, utilities)
                    optimal_score = optimal_scores[criterion]

                    # Nash: allow approximation upwards
                    if j == 4:
                        if float(score) >= float(optimal_score):
                            tie_break_results[i, j] += 1
                    # EF1: allow approximation downwards
                    elif j == 5:
                        if score <= optimal_score:
                            tie_break_results[i, j] += 1
                    # Other: exact match
                    else:
                        if score == optimal_score:
                            tie_break_results[i, j] += 1

        percentage_optimal = (tie_break_results / len(perms)) * 100
        min_percents = np.minimum(min_percents, percentage_optimal)
        max_percents = np.maximum(max_percents, percentage_optimal)

    return min_percents, max_percents


if __name__ == "__main__":
    dic_min = {}
    dic_max = {}

    for alpha in [0.1, 0.3, 0.7, 0.9]:
        total_min = np.zeros((len(algorithms), len(criteria)))
        total_max = np.zeros((len(algorithms), len(criteria)))
        t = 0

        for n in [3, 4]:
            for m in [4, 5, 6]:
                for _ in range(10):
                    utilities = generate_synthetic_utilities(n, m, alpha)
                    min_percent, max_percent = process_instance_with_tiebreak(utilities)
                    total_min += min_percent
                    total_max += max_percent
                    t += 1
        avg_min = total_min / t
        avg_max = total_max / t
        dic_min[alpha] = avg_min
        dic_max[alpha] = avg_max
    ### Spliddit
    directory_path = Path(r'C:\Users\User\Downloads\spliddit')
    file_pattern = directory_path / '*.INSTANCE'
    file_list = glob.glob(str(file_pattern))

 # Initialize arrays to store cumulative min and max percentages across instances
    total_min = np.zeros((len(algorithms), len(criteria)))
    total_max = np.zeros((len(algorithms), len(criteria)))
    t = 0

    for file_path in file_list:
        utilities = np.array(extract_matrix(file_path))+1 
        # to prevent zero utilities (for the nash optimization)
        n, m = utilities.shape
        if (n==3 and m>3 and m<6):
            t += 1
            # Get the minimum and maximum percentages for each algorithm-criterion pair for this instance
            min_percent, max_percent = process_instance_with_tiebreak(utilities)
            
            # Accumulate the min/max percentages across instances
            total_min += min_percent
            total_max += max_percent


    # Average the min/max percentages over all instances
    avg_min_percent = total_min / t if t > 0 else total_min
    avg_max_percent = total_max / t if t > 0 else total_max

    print("Averaged Minimum Percentages (Worst-case Tie-breaking):")
    print(avg_min_percent)

    print("\nAveraged Maximum Percentages (Best-case Tie-breaking):")
    print(avg_max_percent)

