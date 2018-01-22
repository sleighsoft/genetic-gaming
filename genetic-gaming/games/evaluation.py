import multiprocessing
import subprocess
from time import gmtime, strftime

MAX_ROUNDS = 10
MUTATION_MAX = 0.3
MUTATION_STEPS = [x / 10 for x in range(0, int(MUTATION_MAX * 10))]
NUM_NETWORKS = 10
COMMAND_TEMPLATE = "python genetic-gaming/run.py -config genetic-gaming/config/racing.json " \
                   "-max_rounds {max_rounds} " \
                   "-mutation_rate {mutation_rate} " \
                   "-num_networks {num_networks} " \
                   "-save_to {save_to}"


def worker(command):
  process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)
  out, _ = process.communicate()
  return out


if __name__ == '__main__':
  save_to = strftime("%d-%b-%Y_%H:%M:%S", gmtime())
  save_to += "rounds_{max_rounds}_mutation_rate_{mutation_rate}_num_networks_{num_networks}"

  print("Save Directory: {}".format(save_to))
  print("Max Rounds: {}".format(MAX_ROUNDS))
  print("Mutation Steps: {}".format(MUTATION_STEPS))
  print("Num networks: {}".format(NUM_NETWORKS))
  print("----------------")
  PROCESSES = 5
  WORKER_CALLS = 7

  commands = []
  for mutation_rate_extended in MUTATION_STEPS:
    mutation_rate = mutation_rate_extended / 10
    parameter_dict = {'max_rounds': MAX_ROUNDS, 'mutation_rate': mutation_rate, 'num_networks': NUM_NETWORKS}
    save_to_dir = save_to.format(**parameter_dict)
    command = COMMAND_TEMPLATE.format(**parameter_dict, save_to=save_to_dir)
    commands.append(command)

  pool = multiprocessing.Pool(processes=PROCESSES)
  pool_outputs = pool.map(worker, commands)
  pool.close()
  pool.join()
  print('Pool:', pool_outputs)
