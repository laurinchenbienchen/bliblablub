# imports von modulen
import random
import cv2
# imports von funktionen/files
import transformations
import fitness_function
import genetic_operations


# apply transformations and compute fitness
def evaluate_fitness(individual, image1, image2):
    # apply translation, rotattion, scaling based on the individual's parameters
    transformed_image = transformations.translate_image(image2, individual['tx'], individual['ty'])
    transformed_image = transformations.rotate_image(transformed_image, individual['angle'])
    transformed_image = transformations.scale_image(transformed_image, individual['scale'])

    # resize the transformed image to match the size of the reference image
    transformed_image = cv2.resize(transformed_image, (image1.shape[1], image1.shape[0]))

    # ----------------- fitness function anpassen ------- die de-auskommentieren die man gerade haben möchte
    # Compute fitness with mse
    fitness_value = -fitness_function.compute_mse(image1, transformed_image)

    # compute fitness with ssi
    # fitness_value = -fitness_function.fitness_ssim(image1, transformed_image)

    # compute fitness with ncc
    # fitness_value = -fitness_function.fitness_ncc(image1, transformed_image)

    # compute fitness with psnr
    # fitness_value = -fitness_function.fitness_psnr(image1, transformed_image)

    # compute fitness with mi
    # fitness_value = -fitness_function.m_i(image1, image2=transformed_image, bins = 20)
    return fitness_value


# Genetic algorithm
def gen_Algo(image1, image2, population_size=20, generation=50):
    # initialize population with random individuals
    population = [genetic_operations.create_individual() for _ in range(population_size)]

    for gen in range(generation):
        # evaluate fitness of each individual in the population
        fitness_scores = [(ind, evaluate_fitness(ind, image1, image2)) for ind in population]
        # sort individuals based on fitness scores in descending order
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # select best individuals as parents for the next generation
        next_population = [ind for ind, score in fitness_scores[:population_size // 2]]

        # creat new individuals through crossover and mutation
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(next_population, 2)
            child = genetic_operations.crossover(parent1, parent2)
            child = genetic_operations.mutate(child)
            next_population.append(child)

        # update the population for the next generation
        population = next_population

    # return the best individual from the final population
    best_individual = max(population, key=lambda ind: evaluate_fitness(ind, image1, image2))
    return best_individual
