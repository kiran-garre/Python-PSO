import numpy as np
import matplotlib.pyplot as plt
import math


def minimize(f, dim, boundsArr, iterations=50, depth=1, center=[], scale_factor=0.1):
  
    # set bounds and center
    boundsArr = np.array(boundsArr)
    if not len(center):
        center = [0] * dim

    # set number of particles
    number_of_particles = 30

    # create random positions for each particle
    position = np.random.uniform(low=0, high=1, size=(dim, number_of_particles))
    position = position * (boundsArr[:,1][...,None] - boundsArr[:,0][...,None]) + boundsArr[:,0][...,None] # applies bounds to initial positions
    position += np.transpose([center])

    # create array of value of function value at each particle
    values = f(*[*position])

    # create array of best positions for each particle 
    pbest = np.array(position)

    # velocities starts at 0
    velocities = np.zeros((dim, number_of_particles))

    for _ in range(iterations):
        
        # wherever the min value is, save that as global best
        globe = np.array([pbest[:,values.argmin()]])
        
        # parameters for updating velocity
        r1 = np.random.rand()
        r2 = np.random.rand()
        c1 = -2*(_/iterations) + 2.5 # how much the particle's best influences its velocity
        c2 = 2*(_/iterations) + 0.5 # how much the global best influences its velocity
        w = 0.5*(c1 + c2) - 1 # inertia
        
        # update velocities of particles
        velocities = w*velocities + c1*r1*(pbest - position) + c2*r2*(globe.T - position) 
        position = position + velocities

        # update values, pbest, and global best
        # add randomness to particle
        random_position = np.random.uniform(low=0, high=1, size=(dim, number_of_particles//2))
        random_position = random_position * (boundsArr[:,1][...,None] - boundsArr[:,0][...,None]) / 2 + boundsArr[:,0][...,None]
        
        # let half of the swarm converge while the other half keeps exploring
        position[:,number_of_particles//2:] += random_position
        
        # update pbest if new value is less than value
        new_values = f(*[*position])
        pbest = np.where(new_values < values, position, pbest)
        
        # update values 
        values = new_values

    globe = pbest[:,values.argmin()]
    
    # call function recursively to find more exact minimum
    if depth == 1:
        return (*globe, f(*globe))
    else:
        return minimize(f, dim, scale_factor * boundsArr, iterations, depth - 1, globe, scale_factor)


def maximize(f, dim, boundsArr, iterations=50, depth=1, center=[], scale=0.1):
    return minimize(lambda x,y: -f(x,y), dim, boundsArr, iterations, depth, center, scale)

def minimize_package(package, depth):
    return minimize(package.f, package.dim, package.bounds, package.iterations, depth, package.center, package.scale)

def maximize_package(package, depth):
    return maximize(package.f, package.dim, package.bounds, package.iterations, depth, package.center, package.scale)


# Function package for easily optimizing a function several times
class FnPackage:
    def __init__(self, f, dim, boundsArr, iterations=50, center=[], scale_factor=0.1) -> None:
        
        """
        Template for getting started:

            def f(x, y):
                return x + y
            
            dim = 2
            bounds = [
                [0,0]
                [0,0]
            ]
            iterations = 100
            center = (0,0)
            scale_factor = 1

            f_package = FnPackage(f, dim, bounds, iterations, center, scale_factor)

        To minimize the function:

            depth = 1
            pso.minimize_package(f_package, depth)
        
        """
        
        self.f = f
        self.dim = dim
        self.bounds = boundsArr
        self.iterations = iterations
        self.center = center
        self.scale = scale_factor
