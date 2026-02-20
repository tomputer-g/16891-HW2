import copy
import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class KRCBSVertexCollision:
    loc: List[Tuple[int]]
    timestep1: int
    timestep2: int
    a1: int
    a2: int
    
@dataclass
class KRCBSConstraint:
    agent: int
    loc: List[Tuple[int]]
    timestep: int
    
    def to_dict(self):
        return {'agent': self.agent, 'loc': self.loc, 'timestep': self.timestep}

def detect_first_collision_for_path_pair(path1, path2, k) -> KRCBSVertexCollision | None:
    ##############################
    # Return the first collision that occurs between two robot paths (or None if there is no collision)
    # min_t = min(len(path1), len(path2))
    # max_t = max(len(path1), len(path2))
    # for t in range(max_t):
    #     last_valid_t_agent1 = min(len(path1), t)
    #     last_valid_t_agent2 = min(len(path2), t)
    #     if get_location(path1, last_valid_t_agent1) == get_location(path2, last_valid_t_agent2):
    #         return {'loc': [get_location(path1, last_valid_t_agent1)], 'timestep': t}
    
    # Check if any vertex collision (within k timesteps of each other) occurs\
    max_t = max(len(path1), len(path2))
    for t1 in range(max_t):
        t1_valid = min(len(path1), t1)
        for t2_valid in range(max(0, t1-k), min(len(path2), t1+k+1)):
            if get_location(path1, t1_valid) == get_location(path2, t2_valid):
                return KRCBSVertexCollision(loc=[get_location(path1, t1_valid)], timestep1=t1, timestep2=t2_valid, a1=-1, a2=-1)
    
    # for t in range(min_t-1):
    #     if get_location(path1, t) == get_location(path2, t+1) and get_location(path1, t+1) == get_location(path2, t):
    #         return {'loc': [get_location(path1, t), get_location(path1, t+1)], 'timestep': t+1}
    
    return None

def detect_collisions_among_all_paths(paths, k) -> List[KRCBSVertexCollision]:
    ##############################
    # Return a list of first collisions between all robot pairs.
    # A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    # causing the collision, and the timestep at which the collision occurred.
    # You should use your detect_collision function to find a collision between two robots.
    collisions = []
    for agent1 in range(len(paths)):
        for agent2 in range(agent1+1, len(paths)):
            res = detect_first_collision_for_path_pair(paths[agent1], paths[agent2], k)
            if res:
                res.a1 = agent1
                res.a2 = agent2
                collisions.append(res)
    return collisions

def KRCBSSplittingPointConstraints(collision):
    constraint1 = KRCBSConstraint(agent=collision.a1, loc=collision.loc, timestep=collision.timestep1).to_dict()
    constraint2 = KRCBSConstraint(agent=collision.a2, loc=collision.loc, timestep=collision.timestep2).to_dict()
    return [constraint1, constraint2]

class KRCBSSolver(object):
    """The high-level search of K-Robust CBS."""
    def __init__(self, my_map, starts, goals, k=0):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        k           - the parameter for K-Robust CBS
        """

        self.start_time = None
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # Compute heuristics for the low-level search.
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        # The parameter for K-Robust CBS.
        self.k = k

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """
        self.start_time = timer.time()
        
        if self.k == 0:
            raise BaseException("KRCBS with k = 0")

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            _path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if _path is None:
                raise BaseException('No solutions')
            root['paths'].append(_path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'], self.k)
        self.push_node(root)

        while len(self.open_list) > 0:
            P = self.pop_node()
            if len(P['collisions']) == 0:
                self.print_results(P)
                return P['paths']

            # Select a collision (right now, arbitrarily select the first one)
            collision = P['collisions'][0]
            constraints = KRCBSSplittingPointConstraints(collision=collision)
            for constraint in constraints:
                Q = copy.deepcopy(P)
                Q['constraints'].extend([constraint])
                agent = constraint['agent']
                new_path = a_star(
                    self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, Q['constraints']
                )

                if new_path is not None:
                    Q['paths'][agent] = new_path
                    Q['collisions'] = detect_collisions_among_all_paths(Q['paths'], self.k)
                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    self.push_node(Q)
        raise BaseException('No solutions')


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
