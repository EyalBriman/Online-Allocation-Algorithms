import gurobipy as gp
from gurobipy import Model, GRB
import numpy as np




def Nash(demands, supply, C):
    num_agents, num_time_steps = demands.shape
    model = gp.Model("Nash")
    w = model.addVars(num_agents, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=1e-6, ub=1)  # Avoid alpha=0 for log
    z = model.addVars(num_time_steps, num_time_steps, lb=0, name="z")
    X = model.addVars(num_time_steps, lb=0, name="X")
    log_alpha = model.addVars(num_agents, name="log_alpha", lb=-GRB.INFINITY)  # Log of alpha

    # Add logarithmic constraints for each alpha
    for i in range(num_agents):
        model.addGenConstrLog(alpha[i], log_alpha[i], name=f"log_constraint_{i}")

    # Objective: Maximize sum of log(alpha) (equivalent to product of alpha)
    model.setObjective(gp.quicksum(log_alpha[i] for i in range(num_agents)), GRB.MAXIMIZE)

    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")

    for t in range(num_time_steps):
        if C == float('inf'):
            model.addConstr(
                X[t] == supply[t] - gp.quicksum(z[t, t_prime] for t_prime in range(t, num_time_steps)) + gp.quicksum(z[t_prime, t] for t_prime in range(t)),
                name=f"effective_supply_{t}"
            )
        else:
            X[t] = supply[t]
        model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= X[t], name=f"supply_constraint_{t}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal, [alpha[i].X for i in range(num_agents)]
    else:
        print("Optimization did not converge.")
        return None, None




def Util(demands, supply, C):
    num_agents, num_time_steps = demands.shape
    
    model = gp.Model("Utilitarian")
    w = model.addVars(num_agents, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=0, ub=1)
    z = model.addVars(num_time_steps, num_time_steps, lb=0, name="z")
    X = model.addVars(num_time_steps, lb=0, name="X")

    model.setObjective(alpha.sum(), GRB.MAXIMIZE)

    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")

    for t in range(num_time_steps):
        if C == float('inf'):
            model.addConstr(
                X[t] == supply[t] - quicksum(z[t, t_prime] for t_prime in range(t, num_time_steps)) + quicksum(z[t_prime, t] for t_prime in range(t)),
                name=f"effective_supply_{t}"
            )
        else:
            X[t] = supply[t]
        model.addConstr(quicksum(w[i, t] for i in range(num_agents)) <= X[t], name=f"supply_constraint_{t}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal, [alpha[i].X for i in range(num_agents)]
    else:
        print("Optimization did not converge.")
        return None, None

def Egal(demands, supply, C):
    num_agents, num_time_steps = demands.shape
    model = gp.Model("Egalitarian")

    w = model.addVars(num_agents, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=0, ub=1)
    z = model.addVars(num_time_steps, num_time_steps, lb=0, name="z")
    X = model.addVars(num_time_steps, lb=0, name="X")
    alpha_min = model.addVar(name="alpha_min", lb=0, ub=1)

    model.setObjective(alpha_min, GRB.MAXIMIZE)

    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")

    for i in range(num_agents):
        model.addConstr(alpha_min <= alpha[i], name=f"alpha_consistency_{i}")

    for t in range(num_time_steps):
        if C == float('inf'):
            model.addConstr(
                X[t] == supply[t] - quicksum(z[t, t_prime] for t_prime in range(t, num_time_steps)) + quicksum(z[t_prime, t] for t_prime in range(t)),
                name=f"effective_supply_{t}"
            )
        else:
            X[t] = supply[t]
        model.addConstr(quicksum(w[i, t] for i in range(num_agents)) <= X[t], name=f"supply_constraint_{t}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal, [alpha[i].X for i in range(num_agents)]
    else:
        print("Optimization did not converge.")
        return None, None
