# Setup

1. Install conda/miniconda
2. Run `conda env create -f environment.yml`
3. Activate environment (see terminal output for command)
4. Run `python genetic-gaming/run.py -config=genetic-gaming/config/flappybird.json -single_process`

## Saving and Restoring
To simplify reproducing bugs or agent accomplishments, it's possible to save and restore all configuration parameters and learned network weights. This can be achieved using the `-save_to=SAVE_DIR` flag and the `-restore_from=SAVE_DIR` flag, respectively.

## Configuration parameters

It's possible to alter nearly every feature included in this package using configuration files.

### Fitness configuration
Configuration options: 'fitness_mode', 'fitness_function_conf', 'aggregate_maps'

The fitness function that will be used can be set with the 'fitness_mode' parameter. 
The system allows different fitness functions to be used and to be combined with each other:
- 'distance_to_start':
    The absolute distance to the spawn-point of the cars (has no fixed upper limit)
- 'distance_to_end':
    The negated distance to the center of the last segment (with 0 as upper limit)
- 'time':
    The time in seconds the car stayed alive
- 'path':
    The path the car travelled. This calculates how far he got on the track, not how far he drove, to avoid cars driving circles.
- 'fastest':
    The top speed reached by the car.
- 'fastest_average':
    The average speed of the car
- 'close_to_path':
    The average distance to the center of the path throughout the cars life, 
    can be described as how exact he followed the center path denoted by the dots.
- 'mixed':
    A combination of two of the above. Can be configured in the fitness_function_conf group, where func_a and _b denote the
    fitness-functions that should be combined and weight_a and _b the factors. Total fitness is calculated func_a() * weight_a + func_b() * weight_b
- 'fastest_path':
    Divides the path the car travelled ('path') by the time he needed to do so.
    
If 'aggregate_maps' is set to an value greater than one, the fitness will be calculated by using
the sum of the last n rounds instead of only using the last round. This makes sense if the map generator
is configured to create a different map in each round.

## Map configuration
Configuration options: 'map_generator', 'map_seed', 'start_mode', 'randomize_map', 'fix_map_rounds', 'map_generator_conf'

There are two map generators implemented, you can choose which one to use with 'map_generator':
- 'map': Uses a tile based generator, where the generated map is hardcoded. If you want to adjust the generated map take a look in the 
    init_walls_with_map.
- 'random': Generates a random map from the starting point. This map generator randomly chooses
    a target point he aims at, calculates how broad the path to this target point should be,
    generates the walls and repeats the whole process as long as he is able to generate a valid element 
    (no walls outside game area and no walls crossing)

The random map generator uses the python random tools to generate the values,
this means that the generated maps are deterministical if the same seed is used.
The used seed will be printed every time the mapgen is run, and can be fixed
using the 'map_seed' parameter.

For the random map generator there is the 'map_generator_conf' dict, that can be used to tweak the difficulty of
the generated maps. 'min_width' and 'max_width' denote the minimum and maximum width of the end of the path 
segments. 'min_angle' and 'max_angle' denote the minimum and maximum angle between the last and the next segment, 
basically how sharp the turns can be. It does not denote whether the turns will go left or right. A 'min_angle' value of 1 radians
will cause all segments to have an turn of at least 1 radians either left or right in the end. 'min_length' and 'max_length'
denote the minimum and maximum length of a segment. All values are equally distributed.

If 'randomize_map' is set to true, a new map will be generated every round. To do
so, a random seed is chosen in the beginning or the fixed seed will be used. After than,
the seed will be increased by 1 every round to vary the map. This still leads
to the same results when using the same seed. 'fix_map_rounds' can be set to an value
greater than zero, to fix the map for the first n rounds, this reduces the difficulty of the overall game.


'start_mode' describes how the spawnpoints of the cars are chosen. The center spawn point
is in general the same every round, but to be able to distinguish the cars, they are spawned
around this fixed point.
- 'fixed': Spawns all of them exactly at the center spawn point.
- 'random_first': Spawns them randomly around the center spawn point, but spawns them
 on the same point for every round afterwards.
- 'random_each': Reassigns the spawn points around the center spawn point after every round.

