import subprocess
from time import gmtime, strftime

# Fixed Parameters
MAX_ROUNDS = 500
NUM_NETWORKS = 10
MAP_SEED = 171207943139723332376316335113061966300

COMMAND_TEMPLATE_BASE = "python genetic-gaming/run.py -config genetic-gaming/config/racing.json " \
                        "-headless " \
                        "-save_to {save_to}"

COMMAND_TEMPLATE_EXTENDED = COMMAND_TEMPLATE_BASE + "" \
                                                    "-game_seed {map_seed} " \
                                                    "-max_rounds {max_rounds} " \
                                                    "-mutation_rate {mutation_rate} " \
                                                    "-num_networks {num_networks} " \
                                                    "-randomize_map {randomize_map} " \
                                                    "-fix_map_rounds {fix_map_rounds} " \
                                                    "-aggregate_maps {aggregate_maps} "


def worker(command):
  process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)
  out, _ = process.communicate()
  return out


if __name__ == '__main__':
  print("Max Rounds: {}".format(MAX_ROUNDS))
  print("Num networks: {}".format(NUM_NETWORKS))
  print("----------------")
  save_to = strftime("%d-%b-%Y_%H:%M:%S", gmtime())
  save_to += "rounds_{max_rounds}_num_networks_{num_networks}".format(max_rounds=MAX_ROUNDS, num_networks=NUM_NETWORKS)
  print("Save Directory: {}...".format(save_to))
  PROCESSES = 5

  commands = []

  # Test 1: Fixed Map
  params = {"mutation_rate": 0.8}
  save_to_test_1 = save_to + "test1_mutation_rate_{mutation_rate}".format(**params)
  command = COMMAND_TEMPLATE_BASE.format(save_to=save_to_test_1) + " -mutation_rate {mutation_rate}".format(**params)
  commands.append(command)

  # Test 2: Vary map TODO: what other parameter to hand over?
  # Mutation Rates: [0.8]
  params = {"mutation_rate": 0.8}
  save_to_test_2 = save_to + "test2_mutation_rate_{mutation_rate}".format(**params)
  command = COMMAND_TEMPLATE_BASE.format(save_to=save_to_test_2) + " -mutation_rate {mutation_rate}".format(**params)
  commands.append(command)

  # Test 3: Vary map + aggregate over 5 maps
  # Mutation Rates: [0.8]
  params = {"mutation_rate": 0.8, "aggregate_maps": 5}
  save_to_test_3 = save_to + "test3_mutation_rate_{mutation_rate}_aggregate_maps_{aggregate_maps}".format(**params)
  command = COMMAND_TEMPLATE_BASE.format(
    save_to=save_to_test_3) + " -mutation_rate {mutation_rate} -aggregate_maps {aggregate_maps}".format(**params)
  commands.append(command)

  # Test 4: Vary Map
  params = {'randomize_map': True, 'aggregate_maps': 1, 'fix_map_rounds': 50}
  save_to_test_4 = save_to + "test4_aggregate_maps_{aggregate_maps}_randomize_map_{randomize_map}" \
                             "_fix_map_rounds_{fix_map_rounds}".format(**params)
  command = COMMAND_TEMPLATE_BASE.format(
    save_to=save_to_test_4) + " -randomize_map {randomize_map} -aggregate_maps {aggregate_maps} -fix_map_rounds {fix_map_rounds}".format(
    **params)
  commands.append(command)

  # Test 5: Vary Mutation Rates
  params = {'randomize_map': False}
  MUTATION_MAX = 1
  mutation_rates = [x / 10 for x in range(0, int(MUTATION_MAX * 10))]

  for mutation_rate in mutation_rates:
    params = {'max_rounds': MAX_ROUNDS, 'mutation_rate': mutation_rate, 'num_networks': NUM_NETWORKS}
    save_to_test_4 = save_to + "test5_max_rounds_{max_rounds}_mutation_rate_{mutation_rate}" \
                               "_num_networks_{num_networks}".format(**params)
    save_to_dir = save_to.format(**params)
    command = COMMAND_TEMPLATE_BASE.format(
      save_to=save_to_test_4) + " -max_rounds {max_rounds} -mutation_rate {mutation_rate} -num_networks {num_networks}".format(
      **params)
    commands.append(command)

  # Test 6: Dynamic Mutation Rate
  # no mutation rate parameter
  save_to_test_6 = "test6_dynamic_mutation_rate"
  command = COMMAND_TEMPLATE_BASE.format(save_to=save_to_test_6)
  commands.append(command)

  for c in commands:
    print(c)
    # pool = multiprocessing.Pool(processes=PROCESSES)
    # pool_outputs = pool.map(worker, commands)
    # pool.close()
    # pool.join()
    # print('Pool:', pool_outputs)
