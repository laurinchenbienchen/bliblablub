import random


# function to create a new individual with random parameters
def create_individual():
    return {
        'tx': random.uniform(-10, 10),  # translation x
        'ty': random.uniform(-10, 10),  # translation y
        'angle': random.uniform(-180, 180),  # rotation angle
        'scale': random.uniform(0.8, 1.2),  # scaling factor
    }


# function to perform crossover between to parent individuals
def crossover(parent1, parent2):
    child = {}
    # for each key in the parent dictionaries
    for key in parent1.keys():
        # randomly choose to inherit the value from either parent1 or parent2
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


# mutation operation
def mutate(individual):
    mutation_probability = 0.2  # probability of mutation
    # apply mutation with the given prob
    if random.random() < mutation_probability:
        # mutate translation, rotation, scaling parameters with a small random value
        individual['tx'] += random.uniform(-1, 1)
        individual['ty'] += random.uniform(-1, 1)
        individual['angle'] += random.uniform(-1, 1)
        individual['scale'] += random.uniform(0.98, 1.02)
    return individual
