import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List

def hungarian_algorithm(agent_goal_costs: Dict[int, List[int]]):
    """
    :param agent_goal_costs: A dictionary mapping agent ids to a list of costs for each goal.
                            The order of goals is the same for all agents.
    :return: A dictionary mapping agent id to goal id.
    """
    ##############################
    # Implement the Hungarian algorithm.
    # Return the optimal assignment as a dictionary mapping agent id to goal id.
    N = len(agent_goal_costs[0])
    hungarian_cost_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hungarian_cost_mat[i][j] = agent_goal_costs[i][j]
    
    _, col_ind = linear_sum_assignment(hungarian_cost_mat)
    
    
    return_dict = {}
    for i in range(N):
        return_dict[i] = col_ind[i]
    return return_dict
