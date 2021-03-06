import argparse
import json
import subprocess
import atexit
import pprint
import uuid

import os
import tensorflow as tf
import signal
import sys
from games.saver import load_saved_data, save_data, get_current_git_hash


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
      'frames',
      'path',
      'path_end',
      'fastest',
      'fastest_average',
      'close_to_path',
      'composite',
      'divide',
      'multiply'
  ]
  START_MODES = [
      'fixed',
      'random_first',
      'random_each'
  ]


class TFMappings(object):
  ACTIVATIONS = {
      'sigmoid': tf.sigmoid,
      'relu': tf.nn.relu,
      'tanh': tf.tanh
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


class ArgumentValidationException(Exception):
  def __init__(self, *args, **kwargs):
    super().__init__(self, *args, **kwargs)


class Argument(object):

  def __init__(self, name, dtype, description=None, **kwargs):
    """Creates an argument used to validate input data.

    Args:
      name: The name of the argument.
      dtype: The data type of the argument.
      description: (Optional) A description of the argument.
      disable_to_argparse: (Optional) If set, this turns add_to_argparse into a
        no-op for this argument.

    Exclusive Args:
      default: A default value of the argument.
      options: A list of possible argument values.
      function: A function taking the argument value, processing it and
        returning a new argument value.

    NOTE: The unspecified exclusive args will be set to `argparse.SUPPRESS` to
    allow them to be discerned from `None` values.
    """
    if kwargs.get('default') is not None:
      assert not (kwargs.get('options') or kwargs.get('function'))
    elif kwargs.get('options') is not None:
      assert not (kwargs.get('default') or kwargs.get('function'))
    elif kwargs.get('function') is not None:
      assert callable(kwargs.get('function'))
      assert not (kwargs.get('default') or kwargs.get('options'))

    self._name = name
    self._dtype = dtype
    self._description = description

    self._default = kwargs.get('default', argparse.SUPPRESS)
    self._options = kwargs.get('options', argparse.SUPPRESS)
    self._function = kwargs.get('function', argparse.SUPPRESS)
    self.disable_to_argparse = kwargs.get('disable_to_argparse', False)

  @property
  def required(self):
    return self._default is argparse.SUPPRESS

  @property
  def has_options(self):
    return self._options is not argparse.SUPPRESS

  @property
  def has_function(self):
    return self._function is not argparse.SUPPRESS

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  @property
  def description(self):
    return self._description

  @property
  def default(self):
    return self._default

  @property
  def options(self):
    return self._options

  @property
  def function(self):
    return self._function

  def validate(self, value):
    """Validates the given value on this argument."""
    if value is not None and type(value) is not self.dtype:
      raise ArgumentValidationException(
          '{} has to be of type {} but is {}'.format(
              value, type(value), self.dtype))
    if self.default is not argparse.SUPPRESS:
      if value is None:
        value = self.default
    if self.options is not argparse.SUPPRESS:
      if value not in self.options:
        raise ArgumentValidationException(
            '{} not in list of possible options {}'.format(
                value, self.options))
    if self.function is not argparse.SUPPRESS:
      value = self.function(value)
    return value

  def add_to_argparse(self, argument_parser):
    """Adds this argument to the given `ArgumentParser` instance."""
    if not self.disable_to_argparse:
      action = 'store'
      choices = None
      kwargs = {'type': self.dtype}
      kwargs.update({'default': argparse.SUPPRESS})
      if self.dtype == bool:
        kwargs.update({'action': 'store_const'})
        # Remove type, default in case of store action
        kwargs.pop('type', None)
        kwargs.update({'const': not self.default})
      if self.dtype == list:
        action = 'append'
        kwargs.update({'action': action})
      if self.options is not argparse.SUPPRESS:
        choices = self.options
        kwargs.update({'choices': choices})
      argument_parser.add_argument(
          '-' + self.name,
          help=self.description,
          **kwargs
      )
    return argument_parser


class FlagAction(argparse.Action):

  def __init__(self,
               option_strings,
               dest,
               const,
               default=None,
               required=False,
               help=None,
               metavar=None):
    super().__init__(
        option_strings=option_strings,
        dest=dest,
        nargs=0,
        const=argparse.SUPPRESS,
        default=default,
        required=required,
        help=help)

  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, self.const)


class ArgumentValidator(object):

  def __init__(self):
    self.arguments = {}

  def register_parameter(self, name, dtype, description=None, **kwargs):
    """Registers a parameter for validation.

    Args:
      name: The name of the argument.
      dtype: The data type of the argument.
      description: (Optional) A description of the argument.
      disable_to_argparse: (Optional) If set, this turns add_to_argparse into a
        no-op for this argument.

    Exclusive Args:
      default: A default value of the argument.
      options: A list of possible argument values.
      function: A function taking the argument value, processing it and
        returning a new argument value.
    """
    assert name not in self.arguments, \
        '{} already registered!'.format(name)
    argument = Argument(name, dtype, description, **kwargs)
    self.arguments.update({name: argument})

  def validate(self, args):
    """Validates `args` with all reqistered parameters.

    Args:
      args: A dictionary with argument name (key, value) pairs.

    Returns:
      A new dictionary with validated arguments and extra arguments that were
      not included in this validator.

    NOTE: This method runs `sys.exit` in case of missing required parameters
    and prints all registered parameters.
    """
    new_args = {}
    for arg, value in args.items():
      if arg in self.arguments:
        # Registered parameters
        argument = self.arguments[arg]
        try:
          value = argument.validate(value)
        except ArgumentValidationException as exc:
          print('Failed to validate `{}`: {}'.format(arg, exc))
          self.print_registered_parameters_and_exit()
        new_args.update({arg: value})
      else:
        # Extra unchecked parameters
        new_args.update({arg: value})

    for name, argument in self.arguments.items():
      if name not in new_args and argument.required:
        self.print_missing_parameter(name)
        self.print_registered_parameters_and_exit()
      if name not in new_args and argument.default is not argparse.SUPPRESS:
        new_args.update({name: argument.default})

    return new_args

  def print_missing_parameter(self, arg):
    print('`{}` parameter is missing!'.format(arg))

  def print_registered_parameters_and_exit(self):
    self.print_registered_parameters()
    sys.exit()

  def print_registered_parameters(self):
    required = []
    required_with_options = []
    required_with_function = []
    required_with_default = []
    for arg in self.arguments.values():
      if arg.required:
        required.append(arg)
      elif arg.has_options:
        required_with_options.append(arg)
      elif arg.has_function:
        required_with_function.append(arg)
      else:
        required_with_default.append(arg)
    print('Required parameter (name):')
    for arg in required:
      print('\t{}'.format(arg.name))
    print('Required parameter (name -> options):')
    for arg in required_with_options:
      print('\t{} -> {}'.format(arg.name, arg.options))
    print('Required parameter (name -> function):')
    for arg in required_with_function:
      print('\t{} -> {}'.format(arg.name, arg.function.__name__))
    print('Optional parameter (name -> default):')
    for arg in required_with_default:
      print('\t{} -> {}'.format(arg.name, arg.default))

  def add_to_argparse(self, argument_parser=None):
    """Adds all registered arguments to an instance of `ArgumentParser`. This
    does not add parameters with `disable_to_argparse` set to `True`.

    Args:
      argument_parser: (Optional) The argument parser to add the arguments to.
        If none is specified will create a new argument parser.

    Returns:
      An `ArgumentParser`.
    """
    if argument_parser is None:
      argument_parser = argparse.ArgumentParser()
    for argument in self.arguments.values():
      argument_parser = argument.add_to_argparse(argument_parser)
    return argument_parser

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

  @staticmethod
  def validate_map_gen_conf(map_conf_arg):
    map_conf_arg['min_width'] = Argument('min_width', int, 'Minimum track width')\
        .validate(map_conf_arg['min_width'])
    map_conf_arg['max_width'] = Argument('max_width', int, 'Maximum track width')\
        .validate(map_conf_arg['max_width'])
    map_conf_arg['min_angle'] = Argument('min_angle', float, 'Minimum turn angle')\
        .validate(map_conf_arg['min_angle'])
    map_conf_arg['max_angle'] = Argument('max_angle', float, 'Maximum turn angle')\
        .validate(map_conf_arg['max_angle'])
    map_conf_arg['min_length'] = Argument('min_length', int, 'Minimum segment length')\
        .validate(map_conf_arg['min_length'])
    map_conf_arg['max_length'] = Argument('max_length', int, 'Maximum segment length')\
        .validate(map_conf_arg['max_length'])

    return map_conf_arg

  @staticmethod
  def validate_fitness_conf(fitness_conf):
    # Todo: Reimplement validation
    return fitness_conf

  @staticmethod
  def validate_mutation_params(mutation_params):
    mutation_params['c1'] = Argument('c1', float, 'Param C1')\
        .validate(mutation_params['c1'])
    mutation_params['c2'] = Argument('c2', float, 'Param C2')\
        .validate(mutation_params['c2'])
    mutation_params['c3'] = Argument('c3', float, 'Param C3')\
        .validate(mutation_params['c3'])
    return mutation_params


def get_genetic_validator(argument=None):
  """Creates an `ArgumentValidator` for genetic models."""
  if argument is None:
    argument = ArgumentValidator()
  argument.register_parameter(
      'game',
      str,
      'Game to play',
      options=ArgumentConstants.GAMES)
  argument.register_parameter(
      'start_mode',
      str,
      'Mode for generation of starting positions',
      options=ArgumentConstants.START_MODES)
  argument.register_parameter(
      'fitness_mode',
      str,
      'Fitness function to rank networks with',
      options=ArgumentConstants.FITNESS_MODES)
  argument.register_parameter(
      'num_networks',
      int,
      'Number of networks to use')
  argument.register_parameter(
      'num_top_networks_to_keep',
      int,
      'Number of best networks to keep unchanged during evolution')
  argument.register_parameter(
      'num_top_networks_to_mutate',
      int,
      'Number of best networks to mutate during evolution. They will be'
      ' randomly sampled from `num_top_networks_to_keep`. The remaining'
      ' networks will be crossovered'
      ' num_networks - num_top_networks_to_keep - num_top_networks_to_mutate')
  argument.register_parameter(
      'network_input_shape',
      int,
      'Number of inputs of the network')
  argument.register_parameter(
      'mutation_rate',
      float,
      'Regulates the rate of mutation')
  argument.register_parameter(
      'evolve_bias',
      bool,
      'If set, includes the networks biases in the genetic algorithm',
      default=False)
  argument.register_parameter(
      'headless',
      bool,
      'If set, run the game without GUI',
      default=False)
  argument.register_parameter(
      'evolve_kernel',
      bool,
      'If set, includes the networks kernels in the genetic algorithm',
      default=False)
  argument.register_parameter(
      'host',
      str,
      'The host the genetic simulator runs on',
      default='localhost')
  argument.register_parameter(
      'port',
      int,
      'The port the genetic simulator runs on',
      default=4004)
  argument.register_parameter(
      'send_pixels',
      bool,
      'If set, send pixels instead of features to the genetic simulator',
      default=False)
  argument.register_parameter(
      'stepping',
      bool,
      'If set, run one game player after another instead of all '
      'simulataneously',
      default=False)
  argument.register_parameter(
      'single_process',
      bool,
      'If set, run game and network in the same process'
      'simulataneously',
      default=False)
  argument.register_parameter(
      'screen_resize_shape',
      tuple,
      'The resolution the game screen should be resized to. Helps to reduce '
      'network input size',
      default=None)
  argument.register_parameter(
      'tf_seed',
      int,
      'A seed for the tensorflow random number generator',
      default=None)
  argument.register_parameter(
      'game_seed',
      int,
      'A seed for the game\'s random number generator',
      default=None)
  argument.register_parameter(
      'tf_save_model_steps',
      int,
      'The interval (number of evolutions) to save the models progress in',
      default=10)
  argument.register_parameter(
      'aggregate_maps',
      int,
      'The amount of maps that should be taken into account when calculating fitness',
      default=3)
  argument.register_parameter(
      'fix_map_rounds',
      int,
      'The amount of rounds the map should not change in the beginning',
      default=10)
  argument.register_parameter(
      'max_rounds',
      int,
      'Amount of rounds the game is supposed to run. If 0, runs infinitely.',
      default=0)
  argument.register_parameter(
      'randomize_map',
      bool,
      'If true, map seed will be increased by one each round.',
      default=True)
  argument.register_parameter(
      'network_shape',
      list,
      'A list of dictionaries specifying the network shape',
      function=GeneticValidator.validate_network_shape,
      disable_to_argparse=True)
  argument.register_parameter(
      'map_generator_conf',
      dict,
      'A dictionary containing configuration options for the map generator',
      function=GeneticValidator.validate_map_gen_conf,
      disable_to_argparse=True
  )
  argument.register_parameter(
      'fitness_function_conf',
      list,
      'A dictionary containing configuration options for the fitness function',
      function=GeneticValidator.validate_fitness_conf,
      disable_to_argparse=True
  )
  argument.register_parameter(
      'mutation_params',
      dict,
      'A dictionary containing the 3 variables for the mutation generation',
      function=GeneticValidator.validate_mutation_params,
      disable_to_argparse=True
  )
  return argument


def get_general_validator(argument=None):
  """Creates an `ArgumentValidator` for general run settings."""
  if argument is None:
    argument = ArgumentValidator()
  argument.register_parameter(
      'model',
      str,
      'The neural network learning approach',
      options=ArgumentConstants.MODELS)
  return argument


def load_config_and_merge_with_parser(parser_args):
  """Loads settings from `args.config` (a JSON file) and merges them with
  argparse arguments. Settings from `argparse` will override the JSON settings.
  """
  if hasattr(parser_args, 'config'):
    with open(parser_args.config) as config_file:
      merged_args = json.load(config_file)
      for key, value in parser_args.__dict__.items():
        if value is not argparse.SUPPRESS:
          merged_args.update({key: value})
  return merged_args


def merge_config_with_parser_args(config, parser_args):
  for key, value in parser_args.__dict__.items():
    if value is not argparse.SUPPRESS:
      config.update({key: value})
  return config


def run(args):
  if args['model'] == 'genetic':
    return _run_genetic(args)


def _run_genetic(args):
  from models.genetic import EvolutionSimulator
  simulator = EvolutionSimulator(
      args['network_input_shape'],
      args['network_shape'],
      args['num_networks'],
      args['num_top_networks_to_keep'],
      args['num_top_networks_to_mutate'],
      args['mutation_rate'],
      args['evolve_bias'],
      args['evolve_kernel'],
      args['scope'],
      args['save_to'],
      args['tf_save_model_steps'],
      args['mutation_params'],
      seed=args['tf_seed'])
  if args['headless']:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
  if args['single_process']:
    import importlib
    module = importlib.import_module('games.{}.game'.format(args['game']))
    game = module.Game(args, simulator)
    print('Starting game {}'.format(args['game']))
    game.run()
    tf.reset_default_graph()
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


def signal_handler(signal, frame):
  print('Cancelled by Ctrl+C!')
  sys.exit(0)


def restore_config(known_args):
  restore_dir = known_args.restore_from
  save_dir = known_args.save_to
  print('Restoring config from {}'.format(restore_dir))
  try:
    saved_data = load_saved_data(restore_dir)
    version, args = saved_data['version'], saved_data['args']
    if save_dir is not None:
      args['save_dir'] = save_dir  # Keep on saving
    else:
      del args['save_dir']  # Don't overwrite saved state
    current_hash = get_current_git_hash()
    if current_hash != version:
      print('Warning: The saved game was compiled in commit {} '
            'while the current commit is {}.'.format(version, current_hash))
  except ValueError as e:
    print('An error occurred while trying to restore the specified'
          'data: {}'.format(e))
  else:
    return merge_config_with_parser_args(saved_data['args'], known_args)


def create_arg_parse():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      '-config',
      help='Config file name to load run parameters from.',
      type=str,
      required=True
  )
  parser.add_argument(
      '-save_to',
      help='Save whole config, trained networks and seeds to the specified directory.',
      type=str,
      default='./tmp'
  )
  parser.add_argument(
      '-restore_from',
      help='Restore the whole config, trained networks and seeds from the specified directory.',
      type=str
  )
  parser.add_argument(
      '-tf_debug',
      help='Will set tensorflow logging to debug.',
      action='store_true'
  )
  parser.add_argument(
      '--help',
      action='store_true',
      default=argparse.SUPPRESS,
      help='Shows the help message.'
  )
  return parser


def create_arg_parse_for_multi_runner():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      '-config_dir',
      help='Config directory name to load run parameter-files from.',
      type=str,
      required=True
  )
  parser.add_argument(
      '-save_dir',
      help='Directory to save all results.',
      type=str,
      required=True
  )
  parser.add_argument(
      '-tf_debug',
      help='Will set tensorflow logging to debug.',
      action='store_true'
  )
  parser.add_argument(
      '--help',
      action='store_true',
      default=argparse.SUPPRESS,
      help='Shows the help message.'
  )
  return parser


def load_config(parser, args=None):
  general_argument_validator = get_general_validator()
  parser = general_argument_validator.add_to_argparse(parser)
  known_args = parser.parse_known_args(args)[0]
  if known_args.restore_from is not None:
    config = restore_config(known_args)
  else:
    config = load_config_and_merge_with_parser(known_args)
  return general_argument_validator.validate(config), known_args.save_to


def load_genetic_config(validated_config, parser, args=None):
  genetic_argument_validator = get_genetic_validator()
  parser = genetic_argument_validator.add_to_argparse(parser)
  parsed_args = parser.parse_args(args)
  config = merge_config_with_parser_args(validated_config, parsed_args)
  if config.get('help'):
    parser.print_help()
    sys.exit()
  validated_config = genetic_argument_validator.validate(config)

  return validated_config


def run_with_args(args=None):
  parser = create_arg_parse()

  validated_config, save_to = load_config(parser, args)

  if validated_config['model'] == 'genetic':
    validated_config = load_genetic_config(validated_config, parser, args)

  if validated_config['game_seed'] is None:
    validated_config['game_seed'] = uuid.uuid4().int
    print('Setting game_seed to {}'.format(validated_config['game_seed']))
  if validated_config['tf_seed'] is None:
    validated_config['tf_seed'] = uuid.uuid4().int
    print('Setting tf_seed to {}'.format(validated_config['tf_seed']))

  if save_to is not None:
    try:
      save_data(validated_config)
    except ValueError as e:
      print('An error occurred while trying to save the specified'
            'data: {}'.format(e))

  print('Running with the following parameters:')
  pprint.pprint(validated_config)

  if not validated_config['tf_debug']:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)

  signal.signal(signal.SIGINT, signal_handler)

  run(validated_config)


def run_multiple():
  parser = create_arg_parse_for_multi_runner()
  known_args = parser.parse_known_args()[0]

  if not os.path.isdir(known_args.save_dir):
    os.mkdir(known_args.save_dir)

  for f in os.listdir(known_args.config_dir):
    if os.path.isfile(os.path.join(known_args.config_dir, f)):
      run_with_args(['-config', os.path.join(known_args.config_dir, f),
                     '-save_to', os.path.join(known_args.save_dir, f.replace('.json', ''))])


if __name__ == "__main__":
  run_with_args()
