import numpy as np

from libs.gp.math_gp import math_gp
from libs.gp.math_gp import plot_graph

pop_size=50
num_gen=1000
cxpb=0.5
mutpb=0.2
max_depth=5
verbose=True
int_const=[0, 10] #None
float_const=None #[0, 1]
start_depth=[1, 3]
subtree_depth=[1, 2]

test_using_parameters=True

if test_using_parameters:
    def fitness_func(func, x, y):
        return abs(np.pi-func(x, y))

    logbook, best_expr, best_fitness, func = math_gp(fitness_func, num_inputs=2, args_fitness=[1, 2], pop_size=pop_size, num_gen=num_gen, cxpb=cxpb, mutpb=mutpb, max_depth=max_depth, verbose=verbose, int_const=int_const, float_const=float_const, start_depth=start_depth, subtree_depth=start_depth)

    print(best_expr)
    print(best_fitness)
    print(func(1, 2))
    plot_graph(best_expr)

else: # number optimization using trees (don't show good results when compared with other techniques)
    def fitness_func(func):
        return abs(np.pi-func)

    logbook, best_expr, best_fitness, func = math_gp(fitness_func, num_inputs=0, args_fitness=[], pop_size=pop_size, num_gen=num_gen, cxpb=cxpb, mutpb=mutpb, max_depth=max_depth, verbose=verbose, int_const=int_const, float_const=float_const, start_depth=start_depth, subtree_depth=start_depth)

    print(best_expr)
    print(best_fitness)
    print(func)
    plot_graph(best_expr)
