import random
import msgpackrpc
import sys
import time
import tensorflow as tf
import uuid
import os
import math

from functools import reduce


class Network(object):

  def __init__(self, input_shape, network_shape, scope):
    self.scope = scope
    self.initializer = tf.random_uniform_initializer(minval=-0.5, maxval=0.5)
    self.graph = self.build(input_shape, network_shape)
    self.fitness = 0

  def __call__(self, session, x):
    return session.run(self.graph, feed_dict={self.input: x})

  def build(self, input_shape, network_shape):
    with tf.variable_scope(self.scope, initializer=self.initializer) as scope:
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
               num_top_networks_to_keep,
               num_top_networks_to_mutate,
               mutation_rate,
               evolve_bias,
               evolve_kernel,
               scope,
               save_path,
               save_model_steps,
               mut_params,
               weighted_crossover_evolve=False,
               seed=None):
    """Creates an EvolutionSimulator. It provides an easy interface for
    running multiple networks in parallel and improving them through a
    genetic approach.

    Args:
      input_shape: (int) Number of network input nodes.
      network_shape: A list of network shapes.
      num_networks: (int) Number of networks to create.
      num_top_networks_to_keep: (int) Number of best networks to keep when
        evolving.
      num_top_networks_to_mutate: (int)
      mutation_rate: (float) Controls the
      evolve_bias:
      evolve_kernel:
      scope:
      save_path:
      mut_params:
      save_model_steps:
    """
    self.num_top_networks_to_keep = num_top_networks_to_keep
    self.num_top_networks_to_mutate = num_top_networks_to_mutate
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
    self.mutation_rate = mutation_rate
    self.weighted_crossover_evolve = weighted_crossover_evolve
    self.input_shape = input_shape
    self.network_shape = network_shape

    # Set seed
    seed = seed or uuid.uuid4().int
    tf.set_random_seed(seed)
    random.seed(seed)
    print('Tensorflow seed: {}'.format(seed))

    # Implementation of https://blog.openai.com/evolution-strategies/
    self.sigma = 0.2
    self.alpha = 0.7
    if self.weighted_crossover_evolve:
      self.parent_network = Network(input_shape, network_shape,
                                    scope='parent_network')
      self.N = None
    # End Implementation

    # Init networks
    self.networks = self.create_networks(
        self.num_networks, self.input_shape, self.network_shape, self.scope)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(self.save_path, self.session.graph)
    self.saver = tf.train.Saver(save_relative_paths=True)

  def evolve_weighted_crossover(self):
    import numpy as np
    # Parent variables
    parent_vars = self.parent_network.trainable_variables()
    parent_vars_as_vectors = [tf.reshape(v, [-1]) for v in parent_vars]
    vector_sizes = [tf.size(v) for v in parent_vars_as_vectors]
    w = tf.concat(parent_vars_as_vectors, 0)
    # Evolution
    R = [n.fitness for n in self.networks]
    A = tf.constant(
        np.expand_dims((R - np.mean(R)) /
                       (np.std(R) + sys.float_info.epsilon), -1), tf.float32)

    if self.N is None:
      N = tf.random_normal(tf.TensorShape([self.num_networks, w.shape[0]]))
    else:
      N = self.N

    w = w - tf.squeeze(self.alpha / (self.num_networks * self.sigma) *
                       tf.matmul(tf.transpose(N), A))
    new_vars_as_vectors = tf.split(w, vector_sizes, 0)
    new_vars = new_vars = [tf.reshape(n, v.shape)
                           for n, v in zip(new_vars_as_vectors, parent_vars)]
    evolution_ops = []
    for var, new_var in zip(
            self.parent_network.trainable_variables(), new_vars):
      evolution_ops.append(tf.assign(var, new_var))
    # Population Generation
    self.N = tf.random_normal([self.num_networks, w.shape[0].value])
    network_update_ops = []
    for i in range(len(self.networks)):
      w = w + self.sigma * self.N[i, :]
      new_vars_as_vectors = tf.split(w, vector_sizes, 0)
      new_vars = [tf.reshape(n, v.shape) for n, v in zip(
          new_vars_as_vectors, parent_vars)]
      for var, new_var in zip(
              self.networks[i].trainable_variables(), new_vars):
        network_update_ops.append(tf.assign(var, new_var))

    return evolution_ops + network_update_ops

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
    for x, network in zip(inputs, self.networks):
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
    if index < 0 or index > len(self.networks):
      print('[Error] Index {} is not valid'.format(index))
      return False
    return self.networks[index](self.session, input).tolist()

  def calc_unsuccessful_rounds(self, fitnesses):
    avg = reduce(lambda x, y: x + y, fitnesses) / len(fitnesses)
    if self.last_avg_fitness is not None:
      diff = avg / self.last_avg_fitness
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
      for fitness, network in zip(fitnesses, self.networks):
        network.fitness = fitness
      if self.weighted_crossover_evolve:
        evolution = self.evolve_weighted_crossover()
      else:
        evolution = self.evolve_networks()
      self.session.run(evolution)
    print('Evolution took {} seconds!'.format(time.time() - start_time))
    self.current_step += 1
    # if self.save_model_steps > 0:
    #   if self.current_step % self.save_model_steps == 0:
    #     self.save_networks()

    # We reset the default graph to prevent it from blowing up due to
    # progressively more ops being in the graph over time slowing down
    # evolution speed.
    self.save_networks()
    self.session.close()
    tf.reset_default_graph()
    self.session = tf.Session()
    self.writer = tf.summary.FileWriter(self.save_path, self.session.graph)
    self.networks = self.create_networks(
        self.num_networks, self.input_shape, self.network_shape, self.scope)
    self.saver = tf.train.Saver(save_relative_paths=True)
    self.restore_networks()
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
    for n in self.networks:
      n.reset_fitness()

  def _reinitialize_all(self):
    self.session.run([n.reinitialize_network() for n in self.networks])

  def save_networks(self):
    """Saves all networks to `self.checkpoint_path`."""
    print('Saving networks')
    start_time = time.time()
    self.saver.save(self.session, self.checkpoint_path, self.current_step,
                    write_meta_graph=False)
    print('Saving took {} seconds'.format(time.time() - start_time))

  def restore_networks(self, path=None):
    """Restores all networks from the given path, defaults to
    `self.save_path`."""
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
        1. Selecting the best `self.num_top_networks_to_keep` to be unchanged.
        2. Selecting a random network from the best `self.num_top_networks`.
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
    sorted_networks = self.sort_by_fitness(self.networks)
    winners = sorted_networks[0:self.num_top_networks_to_keep]
    if (len(sorted_networks) > 0 and
            len([n for n in self.networks if
                 n.fitness == -sys.maxsize]) == len(self.networks)):
      # Reinitialize all networks, they all failed without any achievement
      print('Resetting all networks as they all have negative fitness!')
      for network in sorted_networks:
        evolution_ops += [network.reinitialize_network()]
    else:
      networks_to_evolve = sorted_networks[self.num_top_networks_to_keep:]
      # Keep num_top_networks_to_keep unchanged
      for i, network in enumerate(networks_to_evolve):
        if i == 0:
          # Crossover of two best networks
          ops = self._perform_crossover(
              network, winners[0], winners[1], self.evolve_bias,
              self.evolve_kernel)
          evolution_ops += ops
        elif i < (len(networks_to_evolve) - self.num_top_networks_to_mutate):
          # Crossover of random winners
          parentA = random.choice(winners)
          parentB = random.choice(winners)
          while parentA == parentB:
            parentB = random.choice(winners)
          ops = self._perform_crossover(
              network, parentA, parentB, self.evolve_bias, self.evolve_kernel)
          evolution_ops += ops
        else:
          # Mutate random winners: num_top_networks_to_mutate
          ops = self.copy_network_variables(random.choice(winners), network)
          evolution_ops += ops
        # Assure, that all assignments are run before performing mutation
        with tf.control_dependencies(evolution_ops):
          ops = self._perform_mutation(
              network, self.evolve_bias, self.evolve_kernel)
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

  def _perform_mutation(self, network, bias, kernel):
    """Perform mutation of `network` with a chance of `mutation_rate`%.

    Args:
      network: The `Network` to which the mutation will be applied.
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
    for v in variables:
      mutation = self._mutate(v)
      mutation_ops.append(tf.assign(v, mutation))
    return mutation_ops

  def get_mut_rate(self):
    # Somehow useful values are returned by:
    # c1 = 0.5, c2 = 1 c3 = 0.1
    if self.mutation_rate is not None:
      return self.mutation_rate
    else:
      return min((1.0,
                  self.mut_params['c1'] * math.exp(
                      -1 / (self.mut_params['c2'] + (self.mut_params['c3'] * self.unsuccessful_rounds)))))

  def _mutate(self, variable):
    """Mutates a `Tensor`.

    Args:
      variable: The variable to mutate.

    Returns:
      A mutated variable. Run `tf.assign(variable, mutated_variable)` to
      apply to the `Tensor`.
    """
    mutation_rate = self.get_mut_rate()
    shape = tf.shape(variable)
    mask = tf.to_float(tf.random_uniform(shape) < mutation_rate)
    mutation = tf.random_uniform(shape) * 0.2 - 0.1
    return variable + mask * mutation

  @staticmethod
  def choose_random(v1, v2):
    return v1 if random.randint(0, 1) == 1 else v2
