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

#### **Boids number impact on performace**:
  - the following graph is based on tests with ```blockSize```==128
  - the number of boids are gradually increasing
  
  
![](https://github.com/LanLou123/Project1-CUDA-Flocking/raw/master/boid1.JPG)

- for some reason I can't get the axis data into graph, here's the data in detail:

|BoidsNum | Naive         | Scattered  grid | Coherent grid  
--------- | ------------- | --------------|------- 
  1000    | 1585.5|	1457.49|	1447.4
  5000    |467.3|	1223.4|	1324.1
  10000   |212.9|	1273.1|	1289.2
  25000   |34.9|	1187.4|	1254.2
  50000   |10.3|	978.4|	1063.9
  75000   |4.1|	738.8|	826.1
  100000   |0|	486.5|	647.2 
  150000   |0|	258.1|	317.7 
  200000   |0|	165.6|	291.7
  250000   |0|	125.6|	232.8
  500000  |0|	36.9|	77.4
  750000   |0|	20.1|	38.3
  1000000  |0|	12.1|	22.3
  2500000  |0|	1.2|	3.1        
  
  
  - Apparently, with the increasing number of boids, all three methods' frame rate will drop. This is simply because increasing number of boids will increase both the sorrounding targets in naive method and the cells number in grid approaches. Also the difference between slope of three lines prove that naive method has the steepest drop due to it's O(n^2) time complexity, and coherent beat scattered as it has coherent memory storage.


#### **blockSize impact on performance**

![](https://github.com/LanLou123/Project1-CUDA-Flocking/raw/master/boid2.JPG)


- According to the graph, there are no much change of all three approaches' framerate, Also, as a common sense, the blocksize should normally be set as a round multiple of the warp size, which is 32 on all current hardware, so I only tested between 32-1024, according to [a stackoverflow question's reply](https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels), it is most prefable to set blocksize between 128-512,however, you would have to try and test yourself to find the best option.


#### **For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

- the answer to this question can be reflected in the first question's graph, it clearly showed that coherent exceeded scattered. As for the reason, as I mentioned in the algorithm introduction part: it's a result of throwing away redundant buffers and accessing coherent memories.


#### **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**

- it is noticable that if we decide to use 27 cells, we then would not need to multiply  maxdistance with 2 like this:``` 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);``` since the target boid is restricted in just one cell, thus cellwidth would become:``` std::max(std::max(rule1Distance, rule2Distance), rule3Distance); ```, therefore, we can think two extreme conditions:
  - when the number of boids is extremly large, since the volumn of two scenarios are 4X4X4 and 3X3X3, 8 cells will have to check more boids each as density are same for both, thus 27 cell might have better performance in this condition.
  - when the number of boids is small, the density of boids will be lower, therefore the disadvantage of looping more cells will be more significant, and 27 cell will correspondingly appear slower.




