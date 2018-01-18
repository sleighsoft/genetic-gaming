import random
import msgpackrpc
import time
import tensorflow as tf
import uuid
import os
import math

from functools import reduce


class Network(object):

  def __init__(self, input_shape, network_shape, scope):
    self.scope = scope
    self.graph = self.build(input_shape, network_shape)
    self.fitness = 0

  def __call__(self, session, x):
    return session.run(self.graph, feed_dict={self.input: x})

  def build(self, input_shape, network_shape):
    with tf.variable_scope(self.scope) as scope:
      self.scope = scope
      layer = tf.placeholder(tf.float32, shape=(None, input_shape),
                             name='input')
      self.input = layer
      for s in network_shape:
        layer = tf.layers.dense(layer, s['shape'], activation=s['activation'],
                                kernel_initializer=s['kernel_initializer'],
                                bias_initializer=s['bias_initializer'],
                                use_bias=s['use_bias'])
    return layer

  @property
  def name(self):
      return self.scope if isinstance(self.scope, str) else self.scope.name

  def trainable_variables(self):
    return tf.trainable_variables(self.scope.name + '/')

  def trainable_kernel(self):
    return [v for v in tf.trainable_variables(self.scope.name + '/')
            if 'kernel' in v.name]

  def trainable_biases(self):
    return [v for v in tf.trainable_variables(self.scope.name + '/')
            if 'bias' in v.name]

  def reset_fitness(self):
    self.fitness = 0

  def reinitialize_network(self):
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name + '/')
    init_op = tf.variables_initializer(variables)
    return init_op


class EvolutionSimulator(object):
  def __init__(self,
               input_shape,
               network_shape,
               num_networks,
               num_top_networks,
               mutation_rate,
               evolve_bias,
               evolve_kernel,
               scope,
               save_path,
               save_model_steps,
               mut_params,
               seed=None):
    """Creates an EvolutionSimulator. It provides an easy interface for
    running multiple networks in parallel and improving them through a
    genetic approach.

    Args:
      input_shape: (int) Number of network input nodes.
      network_shape: A list of network shapes.
      num_networks: (int) Number of networks to create.
      num_top_networks: (int) Number of best networks to keep when evolving.
      mutation_rate: (float) Controls the
      evolve_bias:
      evolve_kernel:
      scope:
      save_path:
      mut_params:
      save_model_steps:
    """

    assert num_networks > 1
    assert num_top_networks > 1
    self.num_top_networks = num_top_networks
    self.mutation_rate = mutation_rate
    self.evolve_kernel = evolve_kernel
    self.evolve_bias = evolve_bias
    self.scope = scope
    self.num_networks = num_networks
    self.save_path = save_path
    self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')
    self.save_model_steps = save_model_steps
    self.current_step = 0
    self.unsuccessful_rounds = 0
    self.last_avg_fitness = None
    self.mut_params = mut_params

    # Set seed
    seed = seed or uuid.uuid4().int
    tf.set_random_seed(seed)
    random.seed(seed)
    print('Tensorflow seed: {}'.format(seed))

    # Init networks
    self.parent_network = Network(input_shape, network_shape, scope='{}Parent'.format(scope))
    self.noise_matrices = []
    for layer_variables in self.parent_network.trainable_variables():
      self.noise_matrices += tf.random_normal(shape=((num_networks,) + tf.shape(layer_variables)), mean=0, stddev=1, dtype=tf.float32)

    self.children_networks = self.create_networks(num_networks, input_shape, network_shape, scope)
    for network_index, network in enumerate(self.children_networks):
      for variable_index, variable in enumerate(network.trainable_variables()):
        variable += self.noise_matrices[variable_index][network_index, :]

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    print(variable.eval(session=self.session))
    self.writer = tf.summary.FileWriter(self.save_path, self.session.graph)
    self.saver = tf.train.Saver(save_relative_paths=True)

  def start_rpc_server(self, host, port):
    """Starts the simulator as an RPC server at `host:port`."""
    self.server = msgpackrpc.Server(self)
    self.server.listen(msgpackrpc.Address(host, port))
    print('Starting simulator server at {}:{}'.format(host, port))
    self.server.start()

  def predict(self, inputs, index=None):
    """Predict outputs for `inputs` using the networks.

    Args:
      inputs: Features of shape `[networks, input_shape]`.
      index: (optional) If set, only take features at `index`.

    Returns:
      Predicted outputs of shape `[networks, num_outputs]` if index is `None`
      otherwise `[num_outputs]`.
    """
    if index is not None:
      return self._predict_single(inputs, index)
    else:
      return self._predict_all(inputs)

  def _predict_all(self, inputs):
    """Get predictions for a all networks.

    Args:
      inputs: Features of shape `[networks, input_shape]`.

    Returns:
      Predicted outputs of shape `[networks, num_outputs]`.
    """
    if len(inputs) != self.num_networks:
      print('[Error] Number of inputs {} does not match number of '
            'networks {}'.format(len(inputs), self.num_networks))
      return False
    predictions = []
    for x, network in zip(inputs, self.children_networks):
      predictions += network(self.session, x).tolist()
    return predictions

  def _predict_single(self, input, index):
    """Get predictions for a single network.

    Args:
      inputs: Features of shape `[input_shape]`.
      index: Index of the network to pick.

    Returns:
      Predicted outputs of shape `[num_outputs]`.
    """
    if index < 0 or index > len(self.children_networks):
      print('[Error] Index {} is not valid'.format(index))
      return False
    return self.children_networks[index](self.session, input).tolist()

  def calc_unsuccessful_rounds(self, fitnesses):
    avg = reduce(lambda x, y: x + y, fitnesses) / len(fitnesses)
    if self.last_avg_fitness is not None:
      diff = avg/self.last_avg_fitness
      if diff < 1.1:
        self.unsuccessful_rounds += 1
      if diff > 1.2:
        self.unsuccessful_rounds = 0
    self.last_avg_fitness = avg

  def evolve(self, fitnesses):
    """Triggers evolution of all networks based on `fitnesses`.

    Args:
      fitnesses: A list of fitnesses where fitnesss[0] corresponds to
        network[0] and so on.
    """
    start_time = time.time()

    self.calc_unsuccessful_rounds(fitnesses)
    if self.unsuccessful_rounds > 50:
      self.reset()
    else:
      for fitness, network in zip(fitnesses, self.children_networks):
        network.fitness = fitness
      evolution = self.evolve_networks()
      self.session.run(evolution)
    print('Evolution took {} seconds!'.format(time.time() - start_time))
    self.current_step += 1
    if self.save_model_steps > 0:
      if self.current_step % self.save_model_steps == 0:
        self.save_networks()
    return True

  def reset(self):
    """Resets all networks. Stored fitnesses are set to 0 and all network
    weights will be initialized again.
    """
    self._reset_fitness_for_all()
    self._reinitialize_all()
    self.current_step = 0
    return True

  def _reset_fitness_for_all(self):
    for n in self.children_networks:
      n.reset_fitness()

  def _reinitialize_all(self):
    self.session.run([n.reinitialize_network() for n in self.children_networks])

  def save_networks(self):
    """Saves all networks to `self.checkpoint_path`."""
    print('Saving networks')
    start_time = time.time()
    self.saver.save(self.session, self.checkpoint_path, self.current_step,
                    write_meta_graph=False)
    print('Saving took {} seconds'.format(time.time() - start_time))

  def restore_networks(self, path=None):
    """Restores all networks from the given path, default to `self.save_path`."""
    if not path:
      path = self.save_path
    latest_checkpoint = tf.train.latest_checkpoint(path)
    if latest_checkpoint:
      self.current_step = int(latest_checkpoint.split('-')[-1])
      print('Restoring networks')
      start_time = time.time()
      self.saver.restore(self.session, latest_checkpoint)
      print('Restoring took {} seconds'.format(time.time() - start_time))

  @staticmethod
  def create_networks(num_networks, input_shape, network_shape, base_scope):
    """Creates `num_networks` distinct networks.

    Args:
      num_networks: Number of networks to create.
      input_shape: Number of network inputs.
      network_shape: A list of dicts specifying the whole network. Dict must
        contain at least `activation`, `shape`, `kernel_initializer`,
        `bias_initializer` and `use_bias`.
      base_scope: A scope name that will be suffixed with the network position.

    Returns:
      A list containing `num_networks` new `Networks`.
    """
    return [Network(input_shape, network_shape,
                    scope='{}{}'.format(base_scope, i))
            for i in range(num_networks)]

  @staticmethod
  def copy_network_variables(source_network, target_network):
    """Assigns all trainable variables of `source_network` to `target_network`.

    Args:
      source_network: A `Network`.
      target_network: A `Network.

    Returns:
      A list of `copy_ops` to be run with `session.run(copy_ops)` to execute
      the copying.
    """
    copy_ops = []
    source_network_variables = source_network.trainable_variables()
    target_network_variables = target_network.trainable_variables()
    for sv, tv in zip(source_network_variables, target_network_variables):
      copy_ops.append(tf.assign(tv, sv))
    return copy_ops


  def evolve_networks(self):
    """Evolves all networks in `self.networks` and uses their individual
    fitnesses to rank them.

    Performs the following operations:
      1. Selection:
        1. Selecting the best `self.num_top_networks` to be unchanged.
        2. Selecting a random network  from the best `self.num_top_networks`.
      2. Crossover:
        1. of the best two networks.
        2. between pairs of random best `self.num_top_networks`.
           For all except the last network.
      3. Mutation: Chance of random mutation of all new networks.

    Returns:
      A list of `evolution_ops` to be run with `session.run(copy_ops)` to
      execute the evolution.
    """
    evolution_ops = []
    sorted_networks = self.sort_by_fitness(self.children_networks)
    winners = sorted_networks[0:self.num_top_networks]
    if len(sorted_networks) > 0 and winners[0].fitness < 0:
      # Reinitialize all networks, they all failed without any achievement
      print('Resetting all networks as they all have negative fitness!')
      for network in sorted_networks:
        evolution_ops += [network.reinitialize_network()]
    else:
      # Keep num_top_networks unchanged
      for i, network in enumerate(sorted_networks[self.num_top_networks:]):
        if i == 0:
          # Network#num_top_networks = Crossover of Winner0 + Winner1
          ops = self._perform_crossover(
              network, winners[0], winners[1], self.evolve_bias,
              self.evolve_kernel)
          evolution_ops += ops
        elif i < len(sorted_networks[self.num_top_networks:]) - 2:
          # Network#num_top_networks+1 to Network#-2 = Crossover of 2 random
          # winners
          parentA = random.choice(winners)
          parentB = random.choice(winners)
          while parentA == parentB:
            parentB = random.choice(winners)
          ops = self._perform_crossover(
              network, parentA, parentB, self.evolve_bias, self.evolve_kernel)
          evolution_ops += ops
        else:
          # Network#last = Random winner
          ops = self.copy_network_variables(random.choice(winners), network)
          evolution_ops += ops
        # Assure, that all assignments are run before performing mutation
        with tf.control_dependencies(evolution_ops):
          ops = self._perform_mutation(
              network, self.mutation_rate, self.evolve_bias,
              self.evolve_kernel)
          evolution_ops += ops
    return evolution_ops

  @staticmethod
  def sort_by_fitness(networks):
    networks = sorted(networks, key=lambda x: x.fitness, reverse=True)
    return networks

  def _perform_crossover(self, crossover_network, network1, network2, bias,
                         kernel):
    """Performs crossover between `network1` and `network2`. The result of the
    crossover will be applied to `crossover_network`. The result is one of
    two possible crossovers at a random "crossover_point":
      1. network1[0:crossover_point] + network2[crossover_point:]
      2. network2[0:crossover_point] + network1[crossover_point:]

    Args:
      crossover_network: The network to which the crossover will be applied.
      network1: A parent `Network`.
      network2: A parent `Network`.
      bias: A `bool`. If `True`, include the bias in the crossover.
      kernel: A `bool`. If `True`, include the kernel in the crossover.

     Returns:
      A list of `crossover_ops` to be run with `session.run(crossover_ops)` to
      execute the crossover on `crossover_network`.
    """
    crossover_ops1 = []
    crossover_ops2 = []
    if bias:
      bias_scope1 = network1.trainable_biases()
      bias_scope2 = network2.trainable_biases()
      bias_crossover = crossover_network.trainable_biases()
      assert len(bias_scope1) == len(bias_scope2), \
          "Number of bias variables was {} for network1 and {} for network2 " \
          "but has to be the same for both networks".format(
          len(bias_scope1), len(bias_scope2))
      assert len(bias_scope1) == len(bias_crossover), \
          "Number of bias variables was {} for network1+2 and {} for " \
          "crossover network but has to be the same for both networks".format(
          len(bias_scope1), len(bias_crossover))
      crossover_point = random.randint(0, len(bias_scope1) - 1)
      for i in range(len(bias_crossover)):
        if i < crossover_point:
          crossover_ops1.append(
              tf.assign(bias_crossover[i], bias_scope1[i]))
          crossover_ops2.append(
              tf.assign(bias_crossover[i], bias_scope2[i]))
        else:
          crossover_ops1.append(
              tf.assign(bias_crossover[i], bias_scope2[i]))
          crossover_ops2.append(
              tf.assign(bias_crossover[i], bias_scope1[i]))
    if kernel:
      kernel_scope1 = network1.trainable_kernel()
      kernel_scope2 = network2.trainable_kernel()
      kernel_crossover = crossover_network.trainable_kernel()
      assert len(kernel_scope1) == len(kernel_scope2), \
          "Number of kernel variables was {} for network1 and {} for " \
          "network2 but has to be the same for both networks".format(
          len(kernel_scope1), len(kernel_scope2))
      assert len(kernel_scope1) == len(kernel_crossover), \
          "Number of kernel variables was {} for network1+2 and {} for " \
          "crossover network but has to be the same for both networks".format(
          len(kernel_scope1), len(kernel_crossover))
      crossover_point = random.randint(0, len(kernel_scope1) - 1)
      for i in range(len(kernel_crossover)):
        if i < crossover_point:
          crossover_ops1.append(
              tf.assign(kernel_crossover[i], kernel_scope1[i]))
          crossover_ops2.append(
              tf.assign(kernel_crossover[i], kernel_scope2[i]))
        else:
          crossover_ops1.append(
              tf.assign(kernel_crossover[i], kernel_scope2[i]))
          crossover_ops2.append(
              tf.assign(kernel_crossover[i], kernel_scope1[i]))

    # Select either A|B or B|A where | indicates the crossover point
    crossover_ops = self.choose_random(crossover_ops1, crossover_ops2)
    return crossover_ops

  def _perform_weighted_crossover(self, crossover_network, network1, network2, bias,
                         kernel):
    """Performs crossover between `network1` and `network2`. The result of the
    crossover will be applied to `crossover_network`. The probability that operation o
    of network n1 will be selected for the offspring is f(n1)/(f(n1)+f(n1)) if n2 is the
    second network and f() yields the fitness of a network.

    Args:
      crossover_network: The network to which the crossover will be applied.
      network1: A parent `Network`.
      network2: A parent `Network`.
      bias: A `bool`. If `True`, include the bias in the crossover.
      kernel: A `bool`. If `True`, include the kernel in the crossover.

     Returns:
      A list of `crossover_ops` to be run with `session.run(crossover_ops)` to
      execute the crossover on `crossover_network`.
    """

    # Probability that a weight from network 1 will be taken
    network1_prob = network1.fitness/(network1.fitness + network2.fitness)

    crossover_ops = []
    if bias:
      bias_scope1 = network1.trainable_biases()
      bias_scope2 = network2.trainable_biases()
      bias_crossover = crossover_network.trainable_biases()
      assert len(bias_scope1) == len(bias_scope2), \
          "Number of bias variables was {} for network1 and {} for network2 " \
          "but has to be the same for both networks".format(
          len(bias_scope1), len(bias_scope2))
      assert len(bias_scope1) == len(bias_crossover), \
          "Number of bias variables was {} for network1+2 and {} for " \
          "crossover network but has to be the same for both networks".format(
          len(bias_scope1), len(bias_crossover))
      # Choose the biases
      for i in range(len(bias_crossover)):
        if random.random() < network1_prob:
          crossover_ops.append(tf.assign(bias_crossover[i], bias_scope1[i]))
        else:
          crossover_ops.append(tf.assign(bias_crossover[i], bias_scope2[i]))
    if kernel:
      kernel_scope1 = network1.trainable_kernel()
      kernel_scope2 = network2.trainable_kernel()
      kernel_crossover = crossover_network.trainable_kernel()
      assert len(kernel_scope1) == len(kernel_scope2), \
          "Number of kernel variables was {} for network1 and {} for " \
          "network2 but has to be the same for both networks".format(
          len(kernel_scope1), len(kernel_scope2))
      assert len(kernel_scope1) == len(kernel_crossover), \
          "Number of kernel variables was {} for network1+2 and {} for " \
          "crossover network but has to be the same for both networks".format(
          len(kernel_scope1), len(kernel_crossover))
      # Choose the kernel
      for i in range(len(kernel_crossover)):
        if random.random() < network1_prob:
          crossover_ops.append(tf.assign(kernel_crossover[i], kernel_scope1[i]))
        else:
          crossover_ops.append(tf.assign(kernel_crossover[i], kernel_scope2[i]))

    return crossover_ops

  def _perform_mutation(self, network, mutation_rate, bias, kernel):
    """Perform mutation of `network` with a chance of `mutation_rate`%.

    Args:
      network: The `Network` to which the mutation will be applied.
      mutation_rate: A `float`. The chance of mutation.
      bias: A `bool`. If `True`, include the bias in the crossover.
      kernel: A `bool`. If `True`, include the kernel in the crossover.

     Returns:
      A list of `mutation_ops` to be run with `session.run(mutation_ops)` to
      execute the mutation on `network`.
    """
    mutation_ops = []
    variables = []
    if kernel:
      variables += network.trainable_kernel()
    if bias:
      variables += network.trainable_biases()
    mutation_rate = self.get_mut_rate()
    for v in variables:
      if random.random() < mutation_rate:
        mutation = self._mutate(v)
        # tf.matmul oder *
        # v_op = tf.matmul(v[10x10], [10x10])
        # tf.assign(v, v_op)
        mutation_ops.append(tf.assign(v, mutation))
    return mutation_ops

  def get_mut_rate(self):
    # Somehow useful values are returned by:
    # c1 = 0.5, c2 = 1 c3 = 0.1
    if self.mut_params['fixed'] is not None:
      return self.mut_params['fixed']
    else:
      return min((1.0,
                  self.mut_params['c1'] * math.exp(
                    -(self.mut_params['c2'] + (self.mut_params['c3'] * self.unsuccessful_rounds)))))

  @staticmethod
  def _mutate(variable):
    """Mutates a `Tensor`self.

    Args:
      variable: The variable to mutate.

    Returns:
      A mutated variable. Run `tf.assign(variable, mutated_variable)` to
      apply to the `Tensor`.
    """
    shape = tf.shape(variable)
    return (variable * (1 + (tf.random_normal(shape) - 0.5) * 3 +
                        tf.random_normal(shape) - 0.5))

  # def get_best_network(networks):
  #   return max(networks, key=lambda x: x.fitness)

  @staticmethod
  def choose_random(v1, v2):
    return v1 if random.randint(0, 1) == 1 else v2
