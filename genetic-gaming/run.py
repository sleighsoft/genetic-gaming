import argparse
import json
import subprocess
import atexit
import pprint
import sys
import tensorflow as tf


game_subprocess = None


@atexit.register
def exit_subprocesses():
  if game_subprocess:
    print('Terminating subprocess: {}'.format(game_subprocess.pid))
    game_subprocess.terminate()


class ArgumentConstants(object):

  MODELS = ['genetic']
  GAMES = ['flappybird', 'racing']
  FITNESS_MODES = [
      'distance_to_start',
      'distance_to_end',
      'time',
      'path',
      'fastest',
      'fastest_average',
      'fastest_average_path']


class TFMappings(object):
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


class ArgumentValidator(object):

  def __init__(self):
    self.registered_parameters = set()
    self.registered_parameters_with_default = {}
    self.registered_parameters_with_fn = {}
    self.registered_parameters_with_options = {}
    # Set containing all parameters registered.
    self.registered_parameter_names = set()
    self.required_parameter_names = set()
    self.optional_parameter_names = set()

  def register_parameter(self, name):
    """Registers a parameter in the list of REQUIRED parameters.

    Args:
      name: Paramater name.

    This is a no_op if `name` is already reqistered.
    """
    if name not in self.registered_parameter_names:
      self.registered_parameter_names.update([name])
      self.required_parameter_names.update([name])
      self.registered_parameters.update([name])
    else:
      print('`{}` already registered!'.format(name))

  def register_parameter_with_default(self, name, default):
    """Registers a parameter in the list of OPTIONAL parameters with a default
    value as fallback. Picks the default if the argument value is `None`.

    Args:
      name: Paramater name.
      default: A default value.

    This is a no_op if `name` is already reqistered.
    """
    if name not in self.registered_parameter_names:
      self.registered_parameter_names.update([name])
      self.optional_parameter_names.update([name])
      self.registered_parameters_with_default.update({name: default})
    else:
      print('`{}` already registered!'.format(name))

  def register_parameter_with_function(self, name, fn):
    """Registers a parameter in the list of REQUIRED parameters with a
    function that takes the argument value as input and returns a new value
    for it e.g. `fn(args[name]) -> new_value`.

    Args:
      name: Paramater name.
      fn: A callable which takes the argument value as input
        and returns a new value.

    This is a no_op if `name` is already reqistered.
    """
    if name not in self.registered_parameter_names:
      assert callable(fn)
      self.registered_parameter_names.update([name])
      self.required_parameter_names.update([name])
      self.registered_parameters_with_fn.update({name: fn})
    else:
      print('`{}` already registered!'.format(name))

  def register_parameter_with_options(self, name, options):
    """Registers a parameter in the list of REQUIRED parameters with a list of
    `options` to check the argument value against.

    Args:
      name: Paramater name.
      options: A list of allowed argument values.

    This is a no_op if `name` is already reqistered.
    """
    if name not in self.registered_parameter_names:
      self.registered_parameter_names.update([name])
      self.required_parameter_names.update([name])
      self.registered_parameters_with_options.update({name: options})
    else:
      print('`{}` already registered!'.format(name))

  def to_argparser(self):
    parser = argparse.ArgumentParser()
    # TODO(julian)

  def validate(self, args):
    """Validates `args` with all reqistered parameters.

    Args:
      args: A dictionary with argument name (key, value) pairs.

    Returns:
      A new dictionary with validated arguments.

    NOTE: This method runs `sys.exit` in case of missing required parameters
    and prints all configured parameters.
    """
    new_args = {}
    for arg, value in args.items():
      if arg in self.registered_parameters_with_default:
        if value is None:
          value = self.registered_parameters_with_default[arg]
        new_args.update({arg: value})
      elif arg in self.registered_parameters_with_options:
        options = self.registered_parameters_with_options[arg]
        if value in options:
          new_args.update({arg: value})
        else:
          self.print_value_not_in_options(arg, value, options)
          self.print_registered_parameters_and_exit()
      elif arg in self.registered_parameters_with_fn:
        value = self.registered_parameters_with_fn[arg](value)
        new_args.update({arg: value})
      elif arg in self.registered_parameters:
        if value is None:
          self.print_missing_parameter(arg)
          self.print_registered_parameters_and_exit()
        else:
          new_args.update({arg: value})
      else:
        print('Extra parameter detected: {} -> {}'.format(arg, value))
        new_args.update({arg: value})

    for arg in self.registered_parameters_with_default:
      if arg not in new_args:
        new_args.update({arg: self.registered_parameters_with_default[arg]})
    for arg in self.required_parameter_names:
      if arg not in new_args:
        self.print_missing_parameter(arg)
        self.print_registered_parameters_and_exit()

    return new_args

  def print_value_not_in_options(self, arg, value, options):
    print('`{}` parameter value `{}` not in `{}`!'.format(arg, value, options))

  def print_missing_parameter(self, arg):
    print('`{}` parameter is missing!'.format(arg))

  def print_registered_parameters_and_exit(self):
    self.print_registered_parameters()
    sys.exit()

  def print_registered_parameters(self):
    print('Required parameter (name):')
    for p in self.registered_parameters:
      print('\t{}'.format(p))
    print('Required parameter (name -> options):')
    for p, options in self.registered_parameters_with_options.items():
      print('\t{} -> {}'.format(p, options))
    print('Required parameter (name -> function):')
    for p, fn in self.registered_parameters_with_fn.items():
      print('\t{} -> {}'.format(p, fn.__name__))
    print('Optional parameter (name -> default):')
    for p, default in self.registered_parameters_with_default.items():
      print('\t{} -> {}'.format(p, default))

  @staticmethod
  def lookup_key_if_exists(dictionary, key, lookup, default=None):
    """If `key` in `dictionary` and `dictionary[key]` in lookup return
    `lookup[dictionary[key]]`, otherwise return `default`.

    Args:
      dictionary: The dictionary to search `key` in.
      key: The key to search for in `dictionary`.
      lookup: A lookup table that maps `dictionary[key]` to a value.
      default: (Optional) A default value to return. Defaults to `None`.
    """
    lookup_value = default
    if key in dictionary:
      if dictionary[key] in lookup:
        lookup_value = lookup[dictionary[key]]
    return lookup_value


class GeneticValidator(object):

  @staticmethod
  def validate_network_shape(network_shape_arg):
    for shape in network_shape_arg:
      shape['activation'] = ArgumentValidator.lookup_key_if_exists(
          shape,
          'activation',
          TFMappings.ACTIVATIONS,
          default=None)
      shape['bias_initializer'] = ArgumentValidator.lookup_key_if_exists(
          shape,
          'bias_initializer',
          TFMappings.INITIALIZER,
          default=None)
      shape['kernel_initializer'] = ArgumentValidator.lookup_key_if_exists(
          shape,
          'kernel_initializer',
          TFMappings.INITIALIZER,
          default=None)
      if 'use_bias' not in shape:
        shape['use_bias'] = False
    return network_shape_arg


def genetic_settings(argument=None):
  """Creates an `ArgumentValidator` for genetic models."""
  if argument is None:
    argument = ArgumentValidator()
  argument.register_parameter_with_options('game', ArgumentConstants.GAMES)
  argument.register_parameter_with_options(
      'fitness_mode', ArgumentConstants.FITNESS_MODES)
  argument.register_parameter('num_top_networks')
  argument.register_parameter('num_top_networks')
  argument.register_parameter('network_input_shape')
  argument.register_parameter('mutation_rate')
  argument.register_parameter('evolve_bias')
  argument.register_parameter('evolve_kernel')
  argument.register_parameter_with_default('host', 'localhost')
  argument.register_parameter_with_default('port', 4004)
  argument.register_parameter_with_default('send_pixels', False)
  argument.register_parameter_with_default('stepping', False)
  argument.register_parameter_with_default('screen_resize_shape', None)
  argument.register_parameter_with_default('save_path', './tmp')
  argument.register_parameter_with_default('stepping', False)
  argument.register_parameter_with_default('tf_seed', None)
  argument.register_parameter_with_default('map_seed', None)
  argument.register_parameter_with_default('tf_save_model_steps', 10)
  argument.register_parameter_with_function(
      'network_shape', GeneticValidator.validate_network_shape)
  return argument


def general_settings(argument=None):
  """Creates an `ArgumentValidator` for general run settings."""
  if argument is None:
    argument = ArgumentValidator()
  argument.register_parameter_with_options('model', ArgumentConstants.MODELS)
  return argument


def load_and_merge_args(args, parser):
  """Loads settings from `args.config` (a JSON file) and merges them with
  argparse arguments. Settings from `argparse` will override the JSON settings.
  """
  if hasattr(args, 'config'):
    with open(args.config) as config_file:
      merged_args = json.load(config_file)
      merged_args.update(args.__dict__)
  return merged_args


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
      args['scope'],
      args['save_path'],
      args['tf_save_model_steps'],
      seed=args['tf_seed'])
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
      required=True
  )
  parser.add_argument(
      '-single_process',
      help='Will run the game with the network in the same process. No'
      ' subprocess will be started to host the game.',
      action='store_true'
  )
  parser.add_argument(
      '-restore_networks',
      help='If set, restore a previously saved model if available.',
      action='store_true'
  )
  parser.add_argument(
      '-tf_debug',
      help='Will set tensorflow logging to debug.',
      action='store_true'
  )
  args = parser.parse_args()
  args = load_and_merge_args(args, parser)

  general_arguments = general_settings()
  validated_args = general_arguments.validate(args)
  if validated_args['model'] == 'genetic':
    genetic_arguments = genetic_settings()
    validated_args = genetic_arguments.validate(validated_args)
  print('Running with the following parameters:')
  pprint.pprint(validated_args)
  if not validated_args['tf_debug']:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)
  run(validated_args)
