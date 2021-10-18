import numpy as np

import operator
from deap import gp
from deap import creator
from deap import base
from deap import tools
from deap import algorithms

import matplotlib.pyplot as plt
import networkx as nx

from libs.gp import operators

def middle_func(individual, pset, fitness_func, args):
    exp = gp.PrimitiveTree(individual)
    exp_str = np.array(list(str(exp)))
    num_exp = 0
    substrings = []
    for a in range(len(exp_str)):
        if exp_str[a] == 'F':
            num_exp += 1
            substrings.append(['_', 'F', '_', '('])
            par_count = 1
            size_sub = 3
            while par_count > 0:
                new_char = exp_str[a+size_sub]
                size_sub += 1
                
                substrings[-1].append(new_char)
                if new_char == ')':
                    par_count -= 1
                elif new_char == '(':
                    par_count += 1
            substrings[-1] = ''.join(substrings[-1])
                    
    if num_exp == 0:
        num_exp += 1
        substrings = [''.join(exp_str)]
    substrings = np.array(substrings)
    functions = [gp.compile(substrings[i], pset) for i in range(len(substrings))]
    #print(substrings)
    return [fitness_func(functions, *args)]


def final_funcs(individual, pset):
    exp = gp.PrimitiveTree(individual)
    exp_str = np.array(list(str(exp)))
    num_exp = 0
    substrings = []
    for a in range(len(exp_str)):
        if exp_str[a] == 'F':
            num_exp += 1
            substrings.append(['_', 'F', '_', '('])
            par_count = 1
            size_sub = 3
            while par_count > 0:
                new_char = exp_str[a+size_sub]
                size_sub += 1
                
                substrings[-1].append(new_char)
                if new_char == ')':
                    par_count -= 1
                elif new_char == '(':
                    par_count += 1
            substrings[-1] = ''.join(substrings[-1])
                    
    if num_exp == 0:
        num_exp += 1
        substrings = [''.join(exp_str)]
    substrings = np.array(substrings)
    return [gp.compile(substrings[i], pset) for i in range(len(substrings))]

def fe_gp(fitness_func, num_inputs, args_fitness=[], pop_size=5, num_gen=100, cxpb=0.5, mutpb=0.2, max_depth=5, verbose=True, int_const=None, float_const=None, start_depth=[1, 3], subtree_depth=[1, 2]):

    # GP operator set
    pset = gp.PrimitiveSet("MAIN", num_inputs)
    pset.addPrimitive(operators._add_, 2)
    pset.addPrimitive(operators._sub_, 2)
    pset.addPrimitive(operators._mul_, 2)
    pset.addPrimitive(operators._div_, 2)
    pset.addPrimitive(operators._neg_, 1) ## FICA?
    pset.addPrimitive(operators._F_, 1)
    pset.addPrimitive(operators._mod_, 1) ## FICA?
    pset.addPrimitive(operators._sqrt_, 1)
    pset.addPrimitive(operators._log_, 1)
    
    if int_const != None:
        pset.addEphemeralConstant('%d_int_const_%d'%(np.random.randint(999999), np.random.randint(999999)), lambda: np.random.randint(*int_const))
    if float_const != None:
        pset.addEphemeralConstant('%d_float_const_%d'%(np.random.randint(999999), np.random.randint(999999)), lambda: np.random.uniform(*float_const))

    # primitive types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # comparation type
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # primitive structure (tree)

    # basic functions
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=start_depth[0], max_=start_depth[1])    # individual create function
    toolbox.register("expr_init", gp.genHalfAndHalf, min_=subtree_depth[0], max_=subtree_depth[1])      # subtree create function
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)                 # create population function

    # GP parameters
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", middle_func, pset=pset, fitness_func=fitness_func, args=args_fitness)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # GP crossover
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = max_depth))
    # GP mutation
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_init, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = max_depth))
    
    # log parameters
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("Q1", np.quantile, q=0.25)
    stats.register("Q2", np.quantile, q=0.5)
    stats.register("Q3", np.quantile, q=0.75)
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(1)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=num_gen, halloffame=hof, stats=stats, verbose=verbose)
    function = final_funcs(gp.PrimitiveTree(hof.items[0]), pset) #gp.compile(gp.PrimitiveTree(hof.items[0]), pset)
    
    return logbook, hof.items[0], hof.keys[0], function

def plot_graph(expr):
    nodes, edges, labels = gp.graph(expr)
    for a in range(len(labels)):
        if labels[a] == '_add_':
            labels[a] = '+'
        elif labels[a] == '_sub_' or labels[a] == '_neg_':
            labels[a] = '-'
        elif labels[a] == '_mul_':
            labels[a] = '*'
        elif labels[a] == '_div_':
            labels[a] = '/'
        elif labels[a] == '_F_':
            labels[a] = 'F'
        elif labels[a] == '_mod_':
            labels[a] = '| |'
        elif labels[a] == '_sqrt_':
            labels[a] = '√'
        elif labels[a] == '_log_':
            labels[a] = 'log'
        elif labels[a] == '_pot_':
            labels[a] = '^'
        elif labels[a] == '_sin_':
            labels[a] = 'sin'
        elif labels[a] == '_cos_':
            labels[a] = 'cos'
        elif str(labels[a])[:3] == 'ARG':
            labels[a] = 'Z[%s]'%labels[a][3]

    g = nx.Graph()

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    node_size=1500
    nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color='#AAAAFF', alpha=1.0, node_shape='o', linewidths=2.0, edgecolors='#000000')
    nx.draw_networkx_edges(g, pos, alpha=1, width=6, edge_color='#FF4444')
    nx.draw_networkx_labels(g, pos, labels, font_size=12, font_color='#000000', font_family='sans-serif', font_weight='normal')
    plt.show()



