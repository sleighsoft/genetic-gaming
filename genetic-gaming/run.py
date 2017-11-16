import argparse
import json
import subprocess
import atexit
import pprint
import tensorflow as tf


game_subprocess = None


@atexit.register
def exit_subprocesses():
  if game_subprocess:
    print('Terminating subprocess: {}'.format(game_subprocess.pid))
    game_subprocess.terminate()


class Validator(object):
  GAMES = ['flappybird', 'racing']
  ACTIVATIONS = {
      'sigmoid': tf.sigmoid,
      'relu': tf.nn.relu
  }
  INITIALIZER = {
      'zeros': tf.zeros_initializer,
      'ones': tf.ones_initializer,
      'constant': tf.constant_initializer,
      'randomUniform': tf.random_uniform_initializer,
      'randomNormal': tf.random_normal_initializer,
      'truncatedNormal': tf.truncated_normal_initializer,
      'uniformUnitScaling': tf.uniform_unit_scaling_initializer,
      'varianceScaling': tf.variance_scaling_initializer,
      'orthogonal': tf.orthogonal_initializer
  }

  def __init__(self, args, parser):
    self.args = args
    self.parser = parser
    self.MODELS = {
        'genetic': self._validate_genetic
    }

  def validate(self):
    self._validate_game()
    self._validate_model()
    return self.args

  def _validate_game(self):
    if 'game' in self.args:
      self._check_parameter('game')
      self._validate_key_in_values(self.args['game'], self.GAMES)

  def _validate_key_in_values(self, key, possible_values):
    if key not in possible_values:
      self.parser.error('{} not in list of possible values {}'.format(
          key, possible_values))

  def _validate_model(self):
    if 'model' in self.args:
      self._check_parameter('model')
      self._validate_key_in_values(self.args['model'], self.MODELS)
      # Run model specific validator
      self.args = self.MODELS[self.args['model']]()

  @staticmethod
  def _lookup_key_if_exists(dictionary, key, lookup, default=None):
    lookup_value = default
    if key in dictionary:
      if dictionary[key] in lookup:
        lookup_value = lookup[dictionary[key]]
    return lookup_value

  @classmethod
  def _validate_network_shape(cls, network_shape):
    for shape in network_shape:
      shape['activation'] = cls._lookup_key_if_exists(
          shape,
          'activation',
          cls.ACTIVATIONS,
          default=None)
      shape['bias_initializer'] = cls._lookup_key_if_exists(
          shape,
          'bias_initializer',
          cls.INITIALIZER,
          default=None)
      shape['kernel_initializer'] = cls._lookup_key_if_exists(
          shape,
          'kernel_initializer',
          cls.INITIALIZER,
          default=None)
      if 'use_bias' not in shape:
        shape['use_bias'] = False
    return network_shape

  def _check_parameter(self, parameter, default=''):
    if parameter not in self.args:
      if default is not '':
        self.args[parameter] = default
        print('Missing `{}` parameter. Default to `{}`'.format(
            parameter, default))
        return True
      self.parser.error('Missing `{}` parameter.'.format(parameter))
    return True

  def _validate_genetic(self):
    self._check_parameter('game')
    self._check_parameter('model')
    self._check_parameter('host', default='localhost')
    self._check_parameter('port', default=4004)
    self._check_parameter('num_networks')
    self._check_parameter('num_top_networks')
    if self._check_parameter('network_shape'):
      args['network_shape'] = self._validate_network_shape(
          args['network_shape'])
    self._check_parameter('network_input_shape')
    self._check_parameter('mutation_rate')
    self._check_parameter('evolve_bias')
    self._check_parameter('evolve_kernel')
    self._check_parameter('scope', default='network')
    self._check_parameter('send_pixels', default=False)
    self._check_parameter('stepping', default=False)
    self._check_parameter('screen_resize_shape', default=None)
    return args


def load_and_merge_args(args, parser):
  if hasattr(args, 'config'):
    with open(args.config) as config_file:
      args = args.__dict__
      args.update(json.load(config_file))
  return args


def run(args):
  if args['model'] == 'genetic':
    _run_genetic(args)


def _run_genetic(args):
  from models.genetic import EvolutionSimulator
  simulator = EvolutionSimulator(
      args['network_input_shape'],
      args['network_shape'],
      args['num_networks'],
      args['num_top_networks'],
      args['mutation_rate'],
      args['evolve_bias'],
      args['evolve_kernel'],
      args['scope'])
  if args['single_process']:
    import importlib
    module = importlib.import_module('games.{}.game'.format(args['game']))
    game = module.Game(args, simulator)
    print('Starting game {}'.format(args['game']))
    game.run()
  else:
    global game_subprocess
    game_path = 'games/{}/game.py'.format(args['game'])
    cmd = ['python', game_path, '-host', args['host'], '-port', args['port'],
           '-birds', args['num_networks']]
    if 'timeout' in args:
      cmd += ['--timeout', args['timeout']]
    cmd = [str(c) for c in cmd]
    print('Starting game {} in subprocess'.format(args['game']))
    game_subprocess = subprocess.Popen(cmd)
    simulator.start_rpc_server(args['host'], args['port'])


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-config',
      help='Config file name to load run parameters from. If specified, all '
      'other CLI arguments will be discarded.',
      type=str,
      default=None,
      required=True
  )
  parser.add_argument(
      '-single_process',
      help='Will run the game with the network in the same process. No'
      ' subprocess will be started to host the game.',
      action='store_true'
  )
  parser.add_argument(
      '-tf_debug',
      help='Will set tensorflow logging to debug.',
      action='store_true'
  )
  args = parser.parse_args()
  args = load_and_merge_args(args, parser)
  v = Validator(args, parser)
  args = v.validate()
  print('Running with the following parameters:')
  pprint.pprint(args)
  if not args['tf_debug']:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)
  run(args)
