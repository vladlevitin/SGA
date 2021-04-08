import random as rd

import matplotlib.pyplot as plt
import numpy as np
import operator
from operator import attrgetter




#use seed to generate the same pseudorandom values
np.random.seed(77)


#a) Implement a function to generate an initial population for your genetic algorithm
#class for each individual in the population

#each Individual is defined as it's own class
class Individual:
    def __init__(self, genome):
        
        self.genome=genome
        self.fitness_val=None


#intialize population with  of random bitstring as the genome
def initialize_population(pop_size, genome_length, fitness_function):
    
    initial_population=[]
    
    for i in range(pop_size):
        
        genome=""
        
        for j in range(genome_length):
            
            genome+=str(rd.randint(0,1))
            
        individual=Individual(genome)
        
        individual.fitness_val=fitness_function(genome)
        
        initial_population.append(individual)
        
        
    
    return initial_population


#Sine fitness function
def fitness_function_sine(genome):
    
        factor=1
        
        if len(genome)>=7:
            factor=2**(len(genome)-7)
        fitness_val=np.sin(int(genome, 2)/factor)
        
        return fitness_val
        
    

#b) Implement a parent selection function for your genetic algorithm. This function should find the fittest individuals in the population, and select parents based on this fitness.

#selects the fittest parents in the top 5% 
def fittest_parent_selection(population, top_precent=0.50):
    
    n=len(population)
    sorted_population=sorted(population, key=lambda x: (x.fitness_val), reverse=True)
    
    #select a parent randomly in the top 10% of the population
    #parent=sorted_population[rd.randint(0,n/20)]
    parent=sorted_population[rd.randint(0,int(n*top_precent))]
    
    return parent
    
    
#randomized parent selection
def randomized_parent_selection(population, n_random=2):
    #the parameters are population and the top  n_parents selected to compete to produce offspring
    
    #sorted_population=sorted(population, key=lambda x: (x.fitness_val), reverse=True)
    
    parents=[]
    
    #competition between n_random parents
    
    for j in range(n_random):
        
        parent=population[rd.randint(0,len(population)-1)]
        
        parents.append(parent)
    
    
    return max(parents, key=attrgetter('fitness_val'))
              
    
#c) Implement a function that creates two offspring from two parents through crossover. The offspring should also have a chance of getting a random mutation. 


def cross_over(parent_a,parent_b, cross_over_rate):
    
    length_a=len(parent_a.genome)
    length_b=len(parent_b.genome)
    
    if length_a!=length_b:
        raise ValueError("Genomes of both parents must be the same length")
        
    if length_a<2:
        return parent_a,parent_b
    
    #the random value is lower than the cross over rate return 
    if(rd.random()<cross_over_rate):
         i=rd.randint(1, length_a-1)
    
         child_a=Individual(parent_a.genome[0:i]+parent_b.genome[i:])
         child_b=Individual(parent_b.genome[0:i]+parent_a.genome[i:])
         
         return child_a, child_b
        
    #else there is no crossover and the function returns children that are the same as their parents
    else:
        return parent_a,parent_b

#mutates the genome based on the mutation rate and number of mutations  
def mutation(genome, n_mutations, mutation_rate):
    for i in range(n_mutations):
            if(rd.random()<mutation_rate):
                j=rd.randint(0,len(genome)-1)
                genome=genome[:j]+str(rd.randint(0,1))+genome[j+1:]
    return genome
        
#combines cross over and mutation function to produce 2 children of 2 parents        
def cross_over_mutation(parent_a, parent_b, fitness_function, cross_over_rate, mutation_rate,n_mutations=1):
    
    
       child_a, child_b=cross_over(parent_a,parent_b, cross_over_rate)
    
       child_a.genome=mutation(child_a.genome, n_mutations, mutation_rate)
    
       child_b.genome=mutation(child_b.genome, n_mutations, mutation_rate)
    
       child_a.fitness_val=fitness_function(child_a.genome)
    
       child_b.fitness_val=fitness_function(child_b.genome)
   
       return child_a, child_b

# d)For a 1-person team: Implement one survivor selection function that selects the survivors of a population based on their fitness

def survivor_selection(new_population, old_population, pop_size):
    
    total_population=new_population+old_population
    
    sorted_population=sorted(total_population, key=lambda x: (x.fitness_val), reverse=True)
    
   
    return sorted_population[0:pop_size]
                       
                       
                       
                    
    
# e) Connect all the implemented functions to complete the genetic algorithm, and run the algorithm with the sine fitness function. throughout the generations plot the individuals, values and fitness values with the sine wave. (2p)    

def generation(old_population, cross_over_rate, mutation_rate, pop_size, fitness_function,parent_selection_function):
    
    new_population=[]
    
    j=1
    
    
    
    while j<=pop_size:
        parent_a=parent_selection_function(old_population)
        parent_b=parent_selection_function(old_population)
        
        child_a, child_b= cross_over_mutation(parent_a,parent_b, fitness_function, cross_over_rate, mutation_rate)
        
        new_population.append(child_a)
        new_population.append(child_b)
        
        j+=2
    #sorted_new_population=sorted(new_population, key=lambda x: (x.fitness_val), reverse=True)    
    
    return new_population

#used to plot the fitness of indiviudals based on the sine fucntion
def plot_data(generations):
    
    x= np.arange(0, 128, 0.001)
    y= np.sin(x)
    
    scaled_genomes=[]
    fitness_values=[]
    
    fig, axs = plt.subplots(len(generations))
  
    
    for i in range(len(generations)):
        for individual in generations[i]:
        
            fitness_values.append(individual.fitness_val)
        

            factor=2**(len(individual.genome)-7)
            scaled_genomes.append(int(individual.genome, 2)/factor)
        
        axs[i].plot(x, y)
        axs[i].scatter(scaled_genomes, fitness_values, color='black')



def simple_genetic_algorithm(pop_size,
        max_gen, 
        cross_over_rate,
        mutation_rate,  
        genome_length, 
        fitness_function,
        parent_selection_function):

    
    
    gen=0
    
    generations=[]
    
    old_population=initialize_population(pop_size, genome_length, fitness_function)
    
    generations.append(old_population)
   
    while gen<max_gen-1:
        
        gen+=1
        
        new_population=generation(old_population, cross_over_rate, mutation_rate, pop_size, fitness_function, parent_selection_function)
        
        old_population=survivor_selection(new_population, old_population, pop_size)
        
        generations.append(old_population)
        
    
    return generations
    #plot_data(generations) #TODO

        
        
#f)Run the genetic algorithm on the provided dataset. Show the results, and compare them to the results of not using any feature selection (given by running the linear regression with all features selected). The points given here depend on the achieved quality of the result and team size. For a 1-person team RMSE less than 0.125. For a 2-person team RMSE less than 0.124



def get_data():
    
    f = open('data.txt','r')
    lines=f.readlines()

    #create an emoty matrix for the data
    data_mat=np.zeros((1994,102))


    #fill in the data values for the code  
    for i in range(len(lines)): 
    
        data=lines[i].split(',')
        data[101]=data[101].replace('\n', '')
    
        for j in range(len(data)):
            data_mat[i][j]=float(data[j])
    #seperate the x values from the y values
    x_val=data_mat[:,:-1]
    y_val=data_mat[:,-1]
    
    return x_val, y_val


import LinReg as lg


#fitness function using linear regression with feature selection

def fitness_function_linreg(genome):
    
    #retrive the data 
    x,y=get_data()
     
    if(len(genome)!=len(x[0])):
        raise ValueError("Genome length has to be the same as x_data")
    
    #create a model linreg
    linreg=lg.LinReg()
    
    #select the important features defined by 1 and remove important features defined by 0
    selected_features=linreg.get_columns(x,genome)
    
    #train the model and return the fitness
    RSME=linreg.get_fitness(selected_features, y)
    
    #return negative as we want to minimize it
    return -RSME
    

#fitness function using linear regression without feature selection

def fitness_function_linreg_all_features(genome):
    x,y=get_data()
     
    if(len(genome)!=len(x[0])):
        raise ValueError("Genome length has to be the same as x_data")
    
    #create a model linreg
    linreg=lg.LinReg()
    
    #train the model and return the fitness
    RSME=linreg.get_fitness(x, y)
    
    #return negative as we want to minimize it
    return -RSME
    

# g) Implement a new survivor selection function. This function should be using a crowding technique as described in the section about crowding. Do exercise f) and g) again with the new selection function, and compare the results to using the simple genetic algorithm. Also show and compare how the entropies of the different approaches (SGA and crowding) change through the generations through a plot. For a 1-person team: implement and demonstrate one crowding approach

#tournament selection/ similar to parent selection in the first part 
"""
def survivor_selection_crowding(new_population, old_population, pop_size, n):
    
    
    total_population=new_population+old_population
    
    survivors=[]
    
    while survivors<pop_size:
        contestants=[]
        for i in range(n):
            contestant=total_population[rd.randint(0,len(total_population))]
            contestants.append(contestant)
            
        survivor=max(contestants, key=attrgetter('fitness_val'))
        
        total_population.remove(survivor)
        
        survivors.append(survivor)
              
    return survivors
"""


#Grouping phase

def hamming_distance(a, b):
    
    if(len(a)!=len(b)):
        raise ValueError("Genomes of both indivividuals")
    
    return sum(c1 != c2 for c1, c2 in zip(a, b))   
    
    

#Crowding phase

#standard crowding algorithm using hamming distances 
def deterministic_crowding(old_population, 
                      pop_size ,
                      cross_over_rate, 
                      mutation_rate, 
                      fitness_function, 
                      parent_selection_function, 
                      distance_function):
    
    survivors=[]
    
    
 
    while (len(survivors)<pop_size):
        
        p1=parent_selection_function(old_population)
        p2=parent_selection_function(old_population)
        
        c1, c2= cross_over_mutation(p1,p2, fitness_function, cross_over_rate, mutation_rate)
        
        if(hamming_distance(p1.genome,c1.genome)+ hamming_distance(p2.genome,c2.genome) < hamming_distance(p1.genome,c2.genome)+ hamming_distance(p2.genome,c1.genome)):
            
            survivor_1=max([p1,c1], key=attrgetter('fitness_val'))
            
            survivor_2=max([p1,c1], key=attrgetter('fitness_val'))
            
        else:
            
            survivor_1=max([p1,c2], key=attrgetter('fitness_val'))
            
            survivor_2=max([p2,c1], key=attrgetter('fitness_val'))
            
        survivors.append(survivor_1)
        survivors.append(survivor_2)

    return survivors

#selects parents completly at random
def select_parents_crowding(population):
    return population[rd.randint(0, len(population)-1)]


def simple_genetic_algorithm_crowding(pop_size,
        max_gen, 
        cross_over_rate,
        mutation_rate,  
        genome_length, 
        fitness_function,
        parent_selection_function):

    
    
    gen=0
    
    generations=[]
    
    old_population=initialize_population(pop_size, genome_length, fitness_function)
    
    generations.append(old_population)
   
    while gen<max_gen-1:
        
        gen+=1
        
        new_population=deterministic_crowding(old_population, 
                      pop_size ,cross_over_rate, 
                      mutation_rate, 
                      fitness_function, 
                      parent_selection_function, 
                      distance_function=hamming_distance)
    
        
        generations.append(new_population)
                      
        old_population=new_population
        
    
    return generations    
    
    
    
    