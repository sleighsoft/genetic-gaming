import os


def get_current_git_hash():
  return os.popen("git log | head -n 1 | sed 's/commit //g' | tr -d '\n'").read()


def build_saver_dict(args):
  current_arguments = args
  current_git_hash = get_current_git_hash()
  data = {'args': current_arguments, 'version': current_git_hash}