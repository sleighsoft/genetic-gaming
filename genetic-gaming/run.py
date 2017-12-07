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
      'fastest_average_path',
      'average_path_distance',
      'close_to_path_with_distance'
  ]


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
      print('\t{}'.format(arg))
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
      'fitness_mode',
      str,
      'Fitness function to rank networks with',
      options=ArgumentConstants.FITNESS_MODES)
  argument.register_parameter(
      'num_networks',
      int,
      'Number of networks to use')
  argument.register_parameter(
      'num_top_networks',
      int,
      'Number of best networks to select from current generation')
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
      'If set, run the game with GUI',
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
      'save_path',
      str,
      'Directory where the model and training information will be saved to',
      default='./tmp')
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
      'network_shape',
      list,
      'A list of dictionaries specifying the network shape',
      function=GeneticValidator.validate_network_shape,
      disable_to_argparse=True)
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


def load_and_merge_args(args, parser):
  """Loads settings from `args.config` (a JSON file) and merges them with
  argparse arguments. Settings from `argparse` will override the JSON settings.
  """
  if hasattr(args, 'config'):
    with open(args.config) as config_file:
      merged_args = json.load(config_file)
      for key, value in args.__dict__.items():
        if value is not argparse.SUPPRESS:
          merged_args.update({key: value})
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
  if args['headless']:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
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
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      '-config',
      help='Config file name to load run parameters from. If specified, all '
      'other CLI arguments will be discarded.',
      type=str,
      required=True
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
  parser.add_argument(
      '--help',
      action='store_true',
      default=argparse.SUPPRESS,
      help='Shows the help message.'
  )
  general_argument_validator = get_general_validator()
  parser = general_argument_validator.add_to_argparse(parser)
  config_arg_position = sys.argv.index('-config')
  config_arg = sys.argv[config_arg_position:config_arg_position + 2]
  args = parser.parse_known_args()[0]
  args = load_and_merge_args(args, parser)
  validated_args = general_argument_validator.validate(args)

  if validated_args['model'] == 'genetic':
    genetic_argument_validator = get_genetic_validator()
    parser = genetic_argument_validator.add_to_argparse(parser)
    args = parser.parse_args()
    args = load_and_merge_args(args, parser)
    if args.get('help'):
      parser.print_help()
      sys.exit()
    validated_args = genetic_argument_validator.validate(args)

  print('Running with the following parameters:')
  pprint.pprint(validated_args)
  if not validated_args['tf_debug']:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)
  run(validated_args)
