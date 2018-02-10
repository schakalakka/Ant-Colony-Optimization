from multiprocessing import Pool
from typing import List, Tuple, Set

import numpy as np


class AntColony:

    def __init__(self, distance_matrix: np.ndarray, nr_ants: int, nr_best: int, nr_iterations: int, decay: float,
                 alpha: float, beta: float, phero_min=0, phero_max=1, nr_procs=1):
        """

        :param distance_matrix: Square distance matrix. Diagonal is not considered
        :param nr_ants: Number of ants per tteration
        :param nr_best: Number of best ants who deposit pheromones
        :param nr_iterations: Number of iterations
        :param decay: Rate of pheromone decay. Should be a number in (0,1) with a low number being a fast decay rate.
        :param alpha: Exponent on pheromone, higher alpha gives pheromone more weight. Default 1
        :param beta: Exponent on distance, higher beta gives distances more weight. Default 1
        :param phero_min: minimum pheromone value
        :param phero_max: maximum pheromone value
        :param nr_procs: grade of parallelism, determines how many paths should be generated in parallel, 1 is minimum
        """
        self.distances = distance_matrix
        self.pheromone = np.ones(self.distances.shape) / len(distance_matrix)
        self.all_indexes = range(len(distance_matrix))
        self.nr_ants = nr_ants
        self.nr_best = nr_best
        self.nr_iterations = nr_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.phero_min = phero_min
        self.phero_max = phero_max
        self.nprocs = max(nr_procs, 1)

    def run(self):
        """
        Main function in class for computing a shortest path.
        :return: List of edges, returns the shortest found path
        """
        # start Pool to be created only once
        with Pool(self.nprocs) as p:
            all_time_shortest_path = ("placeholder", np.inf)
            for i in range(self.nr_iterations):
                all_paths = self.generate_all_paths(p)
                self.update_pheromone(all_paths)
                shortest_path = min(all_paths, key=lambda x: x[1])
                # print(shortest_path)
                if shortest_path[1] < all_time_shortest_path[1]:
                    all_time_shortest_path = shortest_path
            return all_time_shortest_path

    def update_pheromone(self, all_paths: List):
        """
        Updates the pheromone values on the edges.
        POnly the moves from the nr_best paths are considered for updates.
        In the end the pheromone values are trimmed to the phero_min and phero_max value for MIN-MAX-ACO
        :param all_paths:
        :return:
        """
        self.pheromone * self.decay
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:self.nr_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]
        self.pheromone = np.clip(self.pheromone, self.phero_min, self.phero_max)

    def compute_path_distance(self, path: List) -> float:
        """
        Get length of path
        :param path: List of moves through the graph
        :return: float, length of path
        """
        return sum([self.distances[x] for x in path])

    def generate_all_paths(self, p: Pool):
        """
        Generate paths for each ant
        :param p: Pool instance, given as an argument to be created only once
        :return:
        """
        # parallelized version
        all_paths = p.map(self.generate_path, [0] * self.nr_ants)
        all_paths = [(path, self.compute_path_distance(path)) for path in all_paths]
        # sequential version
        # all_paths = []
        # for i in range(self.nr_ants):
        #     path = self.generate_path(0)
        #     all_paths.append((path, self.compute_path_distance(path)))
        return all_paths

    def generate_path(self, start: int) -> List[Tuple[int, int]]:
        """
        Generates a whole graph traversal for one ant
        :param start: starting node, usually 0.
        :return: path, i.e. list of moves
        """
        path = []
        visited = {start}
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone: np.ndarray, dist: np.ndarray, visited: Set) -> int:
        """
        Picks the next edge in a ant traversal
        The current position is given by a row, i.e. pheromones and distances are only one row each
        :param pheromone: row of the whole self.pheromone matrix
        :param dist: row of the whole self.distances matrix
        :param visited: set of already visited nodes
        :return: int, next node in traversal
        """
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(self.all_indexes, 1, p=norm_row)[0]
        return move
