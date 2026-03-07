# Generic imports.
import copy
import time as timer
import heapq
import random
# Project imports.
from hungarian import hungarian_algorithm
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from kr_cbs import detect_first_collision_for_path_pair, detect_collisions_among_all_paths, KRCBSSolver, KRCBSVertexCollision, KRCBSEdgeCollision, KRCBSConstraint


class TACBSSolver(KRCBSSolver):
    """The high-level search of TA-CBS."""
    def __init__(self, my_map, starts, goals, k=0):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        k           - the parameter for K-Robust CBS
        """
        super().__init__(my_map, starts, goals, k)
        self.start_time = None
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        # The parameter for K-Robust CBS.
        self.k = k

    def find_solution(self):
        """
        Finds shortest paths and an optimal target assignment for all agents.
        """
        self.start_time = timer.time()
        # Generate the root node
        # constraints - list of constraints.
        # paths       - list of paths, one for each agent
        #             [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions  - list of collisions in paths.
        # Mc          - Mc[i][j] is the cost of the shortest path (under constraints) for agent i to target j.
        root = {'cost': 0,
                'constraints': [],  # Like in CBS, a list of dictionaries, each dictionary is a constraint.
                'collisions': [],  # Like in CBS.
                'target_assign': [],  # CUSTOM: a copy of the target assignment for this node.
                'paths': [],  # The paths, one for each agent, that are planned for the optimal assignment under Mc.
                'Mc': {i: [float('inf') for g in range(len(self.goals))] for i in range(self.num_of_agents)}
                            # Dict[Int: List[Int]]
                            # Mc[i][j] is the cost of the shortest path (under constraints) for agent i to target j.
                }
        ##############################
        # Find initial paths for each agent to all targets.
        # Populate root['paths'] and root['Mc'] with the paths and costs.
        _root_paths = {}
        for agent_id in range(self.num_of_agents):
            _root_paths[agent_id] = {}
            for goal_id, goal in enumerate(self.goals):
                _root_paths[agent_id][goal_id] = a_star(self.my_map, self.starts[agent_id], goal, self.heuristics[goal_id], agent_id, root['constraints'])
                root['Mc'][agent_id][goal_id] = len(_root_paths[agent_id][goal_id]) if _root_paths[agent_id][goal_id] is not None else float('inf')

        root['target_assign'] = hungarian_algorithm(root['Mc'])
        for agent_id in range(self.num_of_agents):
            goal_id = root['target_assign'][agent_id]
            root['paths'].append(_root_paths[agent_id][goal_id])
            
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'], self.k)
        self.push_node(root)

        ##############################
        # High-Level Search
        
        while len(self.open_list) > 0:
            P = self.pop_node()
            if len(P['collisions']) == 0:
                self.print_results(P)
                return P['paths']
            # Select a collision (right now, arbitrarily select the first one)
            collision = P['collisions'][0]
            for agent_k in [collision.a1, collision.a2]:
                Q = copy.deepcopy(P)
                if type(collision) is KRCBSVertexCollision:
                    Q['constraints'].extend([KRCBSConstraint(agent=agent_k, loc=collision.loc, timestep=(collision.timestep1 if agent_k == collision.a1 else collision.timestep2)).to_dict()])
                elif type(collision) is KRCBSEdgeCollision:
                    Q['constraints'].extend([KRCBSConstraint(agent=agent_k, loc=collision.locs, timestep=(collision.timestep1 if agent_k == collision.a1 else collision.timestep2)).to_dict()])
                else:
                    raise BaseException("Unknown collision type")
                print(Q['constraints'])
                _agent_k_paths = {}
                for goal_id, goal in enumerate(self.goals):
                    _agent_k_paths[goal_id] = a_star(self.my_map, self.starts[agent_k], goal, self.heuristics[goal_id], agent_k, Q['constraints'])
                    Q['Mc'][agent_k][goal_id] = len(_agent_k_paths[goal_id]) if _agent_k_paths[goal_id] is not None else float('inf')
                print(Q['Mc'])
                Q['target_assign'] = hungarian_algorithm(Q['Mc'])
                Q['paths'] = []
                for agent_id in range(self.num_of_agents):
                    goal_id = Q['target_assign'][agent_id]
                    if agent_id == agent_k:
                        Q['paths'].append(_agent_k_paths[goal_id])
                    else:
                    #     Q['paths'].append(P['paths'][agent_id])
                        Q['paths'].append(a_star(self.my_map, self.starts[agent_id], self.goals[goal_id], self.heuristics[goal_id], agent_id, Q['constraints']))
                    
                Q['cost'] = get_sum_of_cost(Q['paths'])
                Q['collisions'] = detect_collisions_among_all_paths(Q['paths'], self.k)
                print(Q)
                self.push_node(Q)
        
        raise BaseException("No Solution")

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
