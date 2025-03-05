import logging
import os
import yaml
import re
import importlib

class Config(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            self.__dict__[key] = value


def convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--") :] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()


def yaml_config_loader(conf_file, overrides=None):
    with open(conf_file, "r") as fr:
        conf_dict = yaml.load(fr, Loader=yaml.FullLoader)
    if overrides is not None:
        overrides = yaml.load(overrides, Loader=yaml.FullLoader)
        conf_dict.update(overrides)
    return conf_dict


def build_config(config_file, overrides=None, copy=False):
    if config_file.endswith(".yaml"):
        if overrides is not None:
            overrides = convert_to_yaml(overrides)
        conf_dict = yaml_config_loader(config_file, overrides)
        if copy and 'exp_dir' in conf_dict:
            os.makedirs(conf_dict['exp_dir'], exist_ok=True)
            saved_path = os.path.join(conf_dict['exp_dir'], 'config.yaml')
            with open(saved_path, 'w') as f:
                f.write(yaml.dump(conf_dict))
    else:
        raise ValueError("Unknown config file format")

    return Config(conf_dict)


def get_logger(fpath=None, fmt=None):
    if fmt is None:
        fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if fpath is not None:
        handler = logging.FileHandler(fpath)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


def dynamic_import(import_path):
    module_name, obj_name = import_path.rsplit('.', 1)
    m = importlib.import_module(module_name)
    return getattr(m, obj_name)

def is_ref_type(value: str):
    assert isinstance(value, str), 'Input value is not a str.'
    if re.match('^<[a-zA-Z]\w*>$', value):
        return True
    else:
        return False

def is_built(ins):
    if isinstance(ins, dict):
        if 'obj' in ins and 'args' in ins:
            return False
        for i in ins.values():
            if not is_built(i):
                return False
    elif isinstance(ins, str):
        if '/' in ins:  # reference may exist in a path string.
            inss = ins.split('/')
            return is_built(inss)
        elif is_ref_type(ins):
            return False
    elif isinstance(ins, list):
        for i in ins:
            if not is_built(i):
                return False
    return True

def deep_build(ins, config, build_space: set = None):
    if is_built(ins):
        return ins

    if build_space is None:
        build_space = set()

    if isinstance(ins, list):
        for i in range(len(ins)):
            ins[i] = deep_build(ins[i], config, build_space)
        return ins
    elif isinstance(ins, dict):
        if 'obj' in ins and 'args' in ins: # return a instantiated module.
            obj = ins['obj']
            args = ins['args']
            assert isinstance(args, dict), f"Args for {obj} must be a dict."
            args = deep_build(args, config, build_space)

            module_cls = dynamic_import(obj)
            mm = module_cls(**args)
            return mm
        else:  # return a nomal dict.
            for k in ins:
                ins[k] = deep_build(ins[k], config, build_space)
            return ins
    elif isinstance(ins, str):
        if '/' in ins:  # reference may exist in a path string.
            inss = ins.split('/')
            inss = deep_build(inss, config, build_space)
            ins = '/'.join(inss)
            return ins
        elif is_ref_type(ins):
            ref = ins[1:-1]

            if ref in build_space:
                raise ValueError("Cross referencing is not allowed in config.")
            build_space.add(ref)

            assert hasattr(config, ref), f"Key name {ins} not found in config."
            attr = getattr(config, ref)
            attr = deep_build(attr, config, build_space)
            setattr(config, ref, attr)

            build_space.remove(ref)
            return attr
        else:
            return ins
    else:
        return ins

def build(name: str, config: Config):
    return deep_build(f"<{name}>", config)

def load_wav_scp(fpath):
    with open(fpath) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {i.split()[0]: i.split()[1] for i in rows}
    return result

