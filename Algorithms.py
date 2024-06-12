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

def max_utility_algorithm(utilities, perm):
    allocation = [[] for _ in range(len(utilities))]
    for j in perm:
        agent = np.argmax([utilities[i, j] for i in range(len(utilities))])
        allocation[agent].append(j)
    return allocation

def most_envy_algorithm(utilities, perm):
    n, m = utilities.shape
    allocation = [[] for _ in range(n)]
    
    for j in perm:
        max_envy = float('-inf')
        most_envy_agent = -1
        
        for i in range(n):
            current_envy = float('-inf')
            for k in range(n):
                if i != k:
                    u_i_S_i = sum(utilities[i, item] for item in allocation[i])
                    u_i_S_k = sum(utilities[i, item] for item in allocation[k])
                    envy_i_k = u_i_S_k + utilities[i, j] - u_i_S_i
                    current_envy = max(current_envy, envy_i_k)
            
            if current_envy > max_envy:
                max_envy = current_envy
                most_envy_agent = i
        allocation[most_envy_agent].append(j)

    return allocation


def least_satisfied_algorithm(utilities, perm):
    allocation = [[] for _ in range(len(utilities))]
    current_utilities = [0] * len(utilities)
    
    for j in perm:
        agents_who_value = [i for i in range(len(utilities)) if utilities[i, j] > 0]
        if agents_who_value:
            least_satisfied_agent = min(agents_who_value, key=lambda i: current_utilities[i])
        else:
            least_satisfied_agent = np.argmin(current_utilities)
        
        allocation[least_satisfied_agent].append(j)
        current_utilities[least_satisfied_agent] += utilities[least_satisfied_agent][j]
    
    return allocation



def nash_product_algorithm(utilities, perm):
    n, m = utilities.shape
    allocation = [[] for _ in range(n)]
    allocated_items = set()  

    for i, j in enumerate(perm):
        if j not in allocated_items:  
            allocation[i % n].append(j)
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

def simulated_annealing(utilities, initial_temp=100, cooling_rate=0.95, steps=5000, restart_probability=0.01):
    n, m = utilities.shape
    allocation = [[] for _ in range(n)]
    items = list(range(m))
    np.random.shuffle(items)
    for i, item in enumerate(items):
        allocation[i % n].append(item)

    current_score = nash_product(allocation, utilities)
    current_allocation = allocation

    temperature = Decimal(initial_temp)

    for step in range(steps):
        if np.random.uniform(0, 1) < restart_probability:
            np.random.shuffle(items)
            allocation = [[] for _ in range(n)]
            for i, item in enumerate(items):
                allocation[i % n].append(item)
            current_allocation = allocation
            current_score = nash_product(current_allocation, utilities)
            temperature = Decimal(initial_temp)
            continue

        new_allocation = [list(bundle) for bundle in current_allocation]
        i, j = np.random.choice(range(n), 2, replace=False)
        if len(new_allocation[i]) > 0 and len(new_allocation[j]) > 0:
            a, b = np.random.choice(new_allocation[i]), np.random.choice(new_allocation[j])
            new_allocation[i].remove(a)
            new_allocation[j].remove(b)
            new_allocation[i].append(b)
            new_allocation[j].append(a)

        all_items = [item for sublist in new_allocation for item in sublist]
        if len(all_items) != len(set(all_items)):
            continue  

        new_score = nash_product(new_allocation, utilities)

        if new_score > current_score or np.random.uniform(0, 1) < np.exp((new_score - current_score) / temperature):
            current_allocation = new_allocation
            current_score = new_score

        temperature *= Decimal(cooling_rate)

    return current_allocation, current_score


def process_instance_file(file_path):
    utilities = np.array(extract_matrix(file_path))
    n, m = utilities.shape
    perms = list(permutations(range(m)))
    score1=[]
    score2=[]
    for perm in perms:
        allocation1 = max_utility_algorithm(utilities, perm)
        allocation2 = least_satisfied_algorithm(utilities, perm)
        score1.append(utilitarian(allocation1, utilities))
        score2.append(egalitarian(allocation2, utilities))
    optimal_scores = []
    optimal_scores.append(max(score1))
    optimal_scores.append(max(score2))
    optimal_allocations = [
        optimal_envy_freeness(utilities),
        optimal_equitability(utilities)
    ]

    optimal_criteria_scores = [envy_freeness, equitability]

    optimal_scores.extend([criterion(optimal_allocations[i], utilities) for i, criterion in enumerate(optimal_criteria_scores)])
    
    sa_allocation = simulated_annealing(utilities)
    optimal_scores.append(nash_product(sa_allocation[0], utilities))

    results = np.zeros((4, 5))
    approximations = np.zeros((4, 5))  
    algorithms = [max_utility_algorithm, least_satisfied_algorithm, most_envy_algorithm, nash_product_algorithm]
    criteria = [utilitarian, egalitarian, envy_freeness, equitability, nash_product]
    for perm in perms:
        for i, algorithm in enumerate(algorithms):
            allocation = algorithm(utilities, perm)
            for j, criterion in enumerate(criteria):
                score = criterion(allocation, utilities)
                if j == 4:  # Nash Product
                    if score >= optimal_scores[j]:
                        results[i, j] += 1
                else:  # Other criteria
                    if score == optimal_scores[3]:
                        results[i, j] += 1
 
                approximations[i, j] += abs(float(score) - float(optimal_scores[j]))

    return results / len(perms) * 100, approximations/len(perms)

def process_all_files(file_list):
    total_results = np.zeros((4, 5))
    total_approximations = np.zeros((4, 5))  
    t = 0
    for file_path in file_list:
        utilities = np.array(extract_matrix(file_path))
        n, m = utilities.shape
        if n <= 8 and m <= 8:
            t += 1
            instance_results, instance_approximations = process_instance_file(file_path) 
            total_results += instance_results
            total_approximations += instance_approximations  
    avg_results = total_results / t if t > 0 else total_results
    avg_approximations = total_approximations / t if t > 0 else total_approximations  
    return avg_results, avg_approximations  

if __name__ == "__main__":
    directory_path = Path(r'PATH')
    file_pattern = directory_path / '*.INSTANCE'
    file_list = glob.glob(str(file_pattern))
    average_results, average_approximations = process_all_files(file_list)
    print("Average Results:")
    print(average_results)
    print("Average Approximations:")
    print(average_approximations)
