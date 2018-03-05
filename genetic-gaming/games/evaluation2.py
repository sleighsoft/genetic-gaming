import multiprocessing
import subprocess
import sys
from time import gmtime, strftime

# Fixed Parameters
N = 10
PARALLELIZE = False
PROCESSES = 5
MAX_ROUNDS = 100
NUM_NETWORKS = 10
DEFAULT_MAP_SEED = 139340055862053188856456977621891175344  # 'Complex 8 shaped'
# DEFAULT_MAP_SEED = 249934098895071520504998917937881504001
COMMAND_TEMPLATE_BASE = "python genetic-gaming/run.py -terminate_if_finished " \
                        "-max_rounds " + str(MAX_ROUNDS) + " -headless " \
                                                           "-num_networks " + str(NUM_NETWORKS) + " " \
                                                                                                  "-save_to {save_to}"
COMMAND_TEMPLATE_EXTENDED = COMMAND_TEMPLATE_BASE + " " \
                                                    "-game_seed {map_seed} " \
                                                    "-config genetic-gaming/config/racing.json"


# "-config genetic-gaming/config/evaluation/{config_file}"


def worker(cmd):
  process = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
  out, _ = process.communicate()
  return out


if __name__ == '__main__':
  args = sys.argv
  if '-parallel' in args:
    PARALLELIZE = True
  print("Running in parallel" if PARALLELIZE else "Running sequentially")
  print("Max Rounds: {}".format(MAX_ROUNDS))
  print("Num networks: {}".format(NUM_NETWORKS))
  print("----------------")
  save_to = strftime("%d-%b-%Y_%H:%M:%S", gmtime())
  save_to += "rounds_{max_rounds}_num_networks_{num_networks}".format(max_rounds=MAX_ROUNDS,
                                                                      num_networks=NUM_NETWORKS)
  print("Save Directory: {}...".format(save_to))

  commands = []
  map_config_files = []
  map_seeds_list = []

  # Map Config 1
  map_config_file = "racing_map_config1.json"
  map_seeds = {
    'Right turn': 162588622003141556655913256491005269949,
    'Long and windy (overall big right turn in the end)': 222823886386910988227991155872541439091,
    'Right turn circle': 320010616754621859186085611061774288661,
    'Huge left turn circle': 110548517694695300904912533898146465388,
    'Straight then right then left': 139340055862053188856456977621891175340,
    'Very small map, left turn': 139340055862053188856456977621891175341,
    'Complex 8 shaped': 139340055862053188856456977621891175344,
    'Very long, mostly right turns': 297905560510116579095157727619522111326,
    'Simple right turn 1': 297905560510116579095157727619522111333,
    'Simple right turn 2': 297905560510116579095157727619522111458,
    'Simple right turn 3': 110857442163165712766571871236463204905,
    'Simple left turn 1': 297905560510116579095157727619522111348,
    'Simple left turn 2': 297905560510116579095157727619522111394,
    'Simple left turn 3': 297905560510116579095157727619522111419,
    'Simple left turn 4': 297905560510116579095157727619522111448,
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 2
  map_config_file = "racing_map_config2.json"
  map_seeds = {
    'Straight but narrow': 274759453521852712689761196754701673871
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 3
  map_config_file = "racing_map_config3.json"
  map_seeds = {
    'Narrow, windy, left turn, right turn': 163920714530746993285169851860080068254,
    'Narrow, one big right turn, small left turn': 163920714530746993285169851860080068260,
    'Narrow, long, right turn, left turn': 31707290214906856065987073907791825371,
    'Narrow, long, simple turns': 31707290214906856065987073907791825376
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 4
  map_config_file = "racing_map_config4.json"
  map_seeds = {
    'Narrow, tough curves': 161058912121400458788655751702753444354,
    'Narrow, long, first right curve, many left curves': 161058912121400458788655751702753444362
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 5
  map_config_file = "racing_map_config5.json"
  map_seeds = {
    'Small segments, narrow, many turns': 336008752973898453599534874846895761267,
    'Small segments, narrow, O shaped': 336008752973898453599534874846895761270
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 6 - straight lines with varying thickness
  map_config_file = "racing_map_config6_straight_lines_varying_thickness.json"
  map_seeds = {
    'Straight lines, varying thickness': 300514681687084028206250151903004404540
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 7 - straight line with same thickness everywhere
  map_config_file = "racing_map_config7_straight_line_same_thickness.json"
  map_seeds = {
    'Straight lines, same thickness': 80753592861910950457841634996230759798
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 8 - straight line with same thickness and funnel in the beginning 1
  map_config_file = "racing_map_config8_straight_line_same_thickness_funnel1.json"
  map_seeds = {
    'Straight lines, same thickness, funnel in beginning 1': 228214018038235441932554259591437865229
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 8 - straight line with same thickness and funnel in the beginning 2
  map_config_file = "racing_map_config8_straight_line_same_thickness_funnel2.json"
  map_seeds = {
    'Straight lines, same thickness, funnel in beginning 2': 179092997821793589553864590502370176300
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 9 - Small paths
  map_config_file = "racing_map_config9_small_paths.json"
  map_seeds = {
    'Long, left turns, right turns': 67243307521636979614083799317130370637,
    'Long, difficult': 67243307521636979614083799317130370640
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  # Map Config 10 - Smooth turns
  map_config_file = "racing_map_config10_smooth_turns.json"
  map_seeds = {
    'Left and right turns': 314553178334753664558439345477832825665,
    'Long': 314553178334753664558439345477832825676
  }
  map_config_files.append(map_config_file)
  map_seeds_list.append(map_seeds)

  ### Evaluation Setup
  # 1) Average over 10 rounds until goal: Vary mutation rate from 0.1 to 0.9 + Dynamic
  if '-mutation_rates' in args:
    for eval_round in range(N):
      for mutation_rate in range(2, 8):
        mutation_rate /= 10.0
        save_to_test_1 = save_to + "evalround_{}_test1_mutation_rate_{}".format(eval_round, mutation_rate)
        command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to_test_1,
                                                   map_seed=DEFAULT_MAP_SEED) + " -num_car_sensors 8 -mutation_rate {}"
        command = command.format(mutation_rate)
        print(command)
        commands.append(command)
      save_to_test_1 = save_to + "evalround_{}_test1_mutation_rate_dynamic".format(eval_round)
      command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to_test_1, map_seed=DEFAULT_MAP_SEED)
      commands.append(command)

  # 2) Number of sensors
  if '-sensors' in args:
    for eval_round in range(N):
      for num_sensor in range(2, 11):
        save_to_test_2 = save_to + "evalround_{}_test2_sensors_{}".format(eval_round, num_sensor)
        command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to_test_2,
                                                   map_seed=DEFAULT_MAP_SEED) + " -num_car_sensors {} -network_input_shape {}"
        command = command.format(num_sensor, num_sensor + 1)
        commands.append(command)

  # 3) Fitness Functions
  if '-fitness_functions' in args:
    for eval_round in range(N):
      for fitness in ['composite', 'path', 'path_end', 'close_to_path']:
        save_to_test_3 = save_to + "evalround_{}_test3_functions_{}".format(eval_round, fitness)
        if fitness == 'composite':
          command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to_test_3,
                                                     map_seed=DEFAULT_MAP_SEED) + " -num_car_sensors 8"
        else:
          command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to_test_3,
                                                     map_seed=DEFAULT_MAP_SEED) + " -num_car_sensors 8 -fitness_mode {}"
          command = command.format(fitness)
        commands.append(command)

  # 3) Fitness Functions
  if '-start_modes' in args:
    for eval_round in range(N):
      for location in ['fixed', 'random_each']:
        save_to_test_4 = save_to + "evalround_{}_test4_location_{}".format(eval_round, location)
        command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to_test_4,
                                                   map_seed=DEFAULT_MAP_SEED) + " -num_car_sensors 8 -start_mode {}"
        command = command.format(location)
        commands.append(command)

  for map_config_file, map_seeds in zip(map_config_files, map_seeds_list):
    for map_description in map_seeds.keys():
      pass
      # save_to_test_10 = save_to + "test10_smooth_turns"
      # command = COMMAND_TEMPLATE_EXTENDED.format(save_to=save_to + map_config_file,
      #                                            config_file=map_config_file, map_seed=map_seeds[map_description])
      # commands.append(command)

      # if PARALLELIZE:
      #     pool = multiprocessing.Pool(processes=PROCESSES)
      #     pool_outputs = pool.map(worker, commands)
      #     pool.close()
      #     pool.join()
      #     print('Pool:', pool_outputs)
      # else:
      #     for c in commands:
      #         print(c)
      #         process = subprocess.Popen(c.split(' '), stdout=subprocess.PIPE)
      #         out, _ = process.communicate()
      #         print(out)

  if PARALLELIZE:
    pool = multiprocessing.Pool(processes=PROCESSES)
    pool_outputs = pool.map(worker, commands)
    pool.close()
    pool.join()
    print('Pool:', pool_outputs)
  else:
    for c in commands:
      print(c)
      process = subprocess.Popen(c.split(' '), stdout=subprocess.PIPE)
      out, _ = process.communicate()
      print(out)
