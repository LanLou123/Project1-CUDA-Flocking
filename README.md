**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Lan Lou
* Tested on: (TODO) Windows 10, i7-6700HQ @ 2.6GHz 16GB, GTX 1070 8GB (personal laptop)

## Coherent grid simulation result

- The simulation shown in the gif was conducted using ```N_FOR_VIS``` = 40k boids agents, and using a uniform coherent grid structure.
  
  
![](https://github.com/LanLou123/Project1-CUDA-Flocking/raw/master/boid0.gif)

## Algorithm description:

- The aim of this project is to simulate birds' flocking behaviour, and use cuda kernals to improve the simulation performance. Boid(bird-oid object) simulation research was originally conducted by Craig Reynolds, in general, his theory tell us three rules a single bird would have to follow in order to achieve the correct result: 
  - cohesion - boids move towards the perceived center of mass of their neighbors.
  - separation - boids avoid getting to close to their neighbors.
  - alignment - boids generally try to move with the same direction and speed as their neighbors.
- guided by these rules, we can build algorithm describing these rules in the form of iterating through a target bird's neighbor, and apply corresponding changes to either the bird's position, or velocity...

- in the project, you will notice that there are three different realization of the simulation algorithms:
  - the first one is called naive: just as it's name implys, it's a brute force method, what actually happend in it is simply like this: for every boid in scope: do a iteration of all other boids and compute the velocity change using those relationships, it neither considered the unnecessity of doing iteration over all boids nor took advantage of cuda's parallel computing.
  - the second is called scattered uniform grid: in this part, we introduced a uniform grid as a checker: the size of the grid is twice the size of the maxium radius of three rule distances. This time, instead of checking all other boids for the target boid, we only check those inside a ```2X2X2``` (or ```3X3X3```) grid depending on the position of the target boid.
  - the third one is called coherent uniform grid: the difference between this and the former one is the way the data is arranged, to be more specific, in the former one, we would first have ```dev_particleArrayIndices``` which stores the position and velocity data index of boids , and ```dev_particleGridIndices``` stores the key binds the grid index to the particle index, actually ```dev_particleArrayIndices``` act as a "pointer" from  ```dev_particleGridIndices``` value to the index of data like position and velocity, after we sorted the data according to the key of  ```dev_particleGridIndices``` we use this "pointer" to access the data of boid, just like this ```vec3 boid_pos = pos[particleArrayIndices[index]];```, however, this is in fact uneccessary, instead of using an additional buffer to access the data, we can directly reshuffle the order of the data into the order of the sorted index, and use this new buffer do direct data accessing like this```vec3 boid_pos = pos_reshuffle[index];```,  there is one huge advantage to this optimization: we won't need additional buffer for extra finding of data, the memory become contiguous making it's calls less.

## Performance Analysis
Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
