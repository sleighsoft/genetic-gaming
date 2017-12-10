import json
import os


def get_current_git_hash():
    return os.popen("git log | head -n 1 | sed 's/commit //g' | tr -d '\n'").read()


def save_data(args):
    save_dir = args['save_to']
    current_arguments = args
    current_git_hash = get_current_git_hash()
    data = {'args': current_arguments, 'version': current_git_hash}
    with open(os.path.join(save_dir, "data.json"), 'w+') as f:
        f.write(json.dumps(data))


def load_saved_data(restore_directory):
    if not os.path.isdir(restore_directory):
        raise ValueError("Directory that should be restored does not exist.")

    data_path = os.path.join(restore_directory, "data.json")
    if not os.path.isfile(data_path):
        raise ValueError("{} should be restored but does not exist".format(data_path))
    with open(data_path) as f:
        data = f.read()
    return json.loads(data)
