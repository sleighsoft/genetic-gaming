import os
from time import gmtime, strftime

MAX_ROUNDS = 10
MUTATION_MAX = 1
MUTATION_STEPS = [x / 10 for x in range(0, int(MUTATION_MAX * 10))]
NUM_NETWORKS = 10
COMMAND_TEMPLATE = "python genetic-gaming/run.py -config genetic-gaming/config/racing.json " \
                   "-max_rounds {max_rounds} " \
                   "-mutation_rate {mutation_rate} " \
                   "-num_networks {num_networks} " \
                   "-save_to {save_to}"

if __name__ == '__main__':
  save_to = strftime("%d-%b-%Y_%H:%M:%S", gmtime())
  os.mkdir(save_to)

  print("Save Directory: {}".format(save_to))
  print("Max Rounds: {}".format(MAX_ROUNDS))
  print("Mutation Steps: {}".format(MUTATION_STEPS))
  print("Num networks: {}".format(NUM_NETWORKS))

  for mutation_rate_extended in MUTATION_STEPS:
    mutation_rate = mutation_rate_extended / 10
    command = COMMAND_TEMPLATE.format(max_rounds=MAX_ROUNDS, mutation_rate=mutation_rate, num_networks=NUM_NETWORKS,
                                      save_to=save_to)
    print(command)
