
import numpy as np
import pandas as pd

from antcolony import AntColony

# read csv and prepare to be a normal numpy distance matrix
distances = pd.read_csv('europe.csv', delimiter=',')
distances = distances.replace('-', np.inf)  # diagonal should be infinity
distances = distances.apply(pd.to_numeric, errors='coerce')  # convert strings in dataframe to floats
distances = distances.values[:, 1:]  # take only the columns and rows with values, i.e. remove city name column


ant_colony = AntColony(distances, nr_ants=20, nr_best=10, nr_iterations=500, decay=0.95,
                       alpha=1, beta=1, phero_min=0.1,
                       phero_max=1, nr_procs=6)
shortest_path = ant_colony.run()
print(f"shortest_path: {shortest_path}")



