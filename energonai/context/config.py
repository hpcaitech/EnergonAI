#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect
import sys
import torch
from typing import Union
from importlib.machinery import SourceFileLoader
from pathlib import Path
from energonai.logging import get_dist_logger


nec_args = {
        'model_class': None,
        'model_type': None,
        'max_batch_size': 32,
        'tp_init_size': 1,
        'pp_init_size': 1,
        'host': "127.0.0.1",
        'port': 29500,
        'dtype': torch.float,
        'checkpoint': None,
        'tokenizer_path': None,
        'server_host': "127.0.0.1",
        'server_port': 8005,
        'log_level': "critical",
        'backend':"nccl",
        'rm_padding': False,
        'seed' : 1024,
        'verbose' : True,
        'trt_sample' : None
}


class Config(dict):
    """This is a wrapper class for dict objects so that values of which can be
    accessed as attributes.

    :param config: The dict object to be wrapped
    :type config: dict
    """

    def __init__(self, config: dict = None):
        if config is not None:
            for k, v in config.items():
                self._add_item(k, v)

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super(Config, self).__getitem__(key)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        super(Config, self).__setitem__(key, value)

    def _add_item(self, key, value):
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def update(self, config):
        assert isinstance(config, (Config, dict)), 'can only update dictionary or Config objects.'
        for k, v in config.items():
            self._add_item(k, v)
        return self

    @staticmethod
    def from_file(filename: str):
        """Reads a python file and constructs a corresponding :class:`Config` object.

        :param filename: Name of the file to construct the return object
        :type filename: str
        :raises AssertionError: Raises an AssertionError if the file does not exist, or the file
            is not .py file
        :return: A :class:`Config` object constructed with information in the file
        :rtype: :class:`Config`
        """

        # check config path
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()

        assert filepath.exists(), f'{filename} is not found, please check your configuration path'

        # check extension
        extension = filepath.suffix
        assert extension == '.py', 'only .py files are supported'

        # import the config as module
        remove_path = False
        if filepath.parent not in sys.path:
            sys.path.insert(0, (filepath))
            remove_path = True

        module_name = filepath.stem
        source_file = SourceFileLoader(fullname=str(module_name), path=str(filepath))
        module = source_file.load_module()

        # load into config
        config = Config()

        for k, v in module.__dict__.items():
            if k.startswith('__') or inspect.ismodule(v) or inspect.isclass(v):
                continue
            else:
                config._add_item(k, v)

        logger = get_dist_logger()
        logger.debug('variables which starts with __, is a module or class declaration are omitted in config file')

        # remove module
        del sys.modules[module_name]
        if remove_path:
            sys.path.pop(0)

        return config


class ConfigException(Exception):
    pass

from colossalai.context.singleton_meta import SingletonMeta

class MetaConfig(metaclass=SingletonMeta):
    def __init__(self):
        self._config = None

    @property
    def config(self):
        return self._config
    
    def __iter__(self):
        return self._config.__iter__()

    def __getitem__(self, key):
        if key in self._config.keys():
            return self._config[key]
        else:
            return None
    
    def __setitem__(self, key, value):
        self._config[key] = value

    def load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dict or a file.
        Args:
            config (dict or str): Either a dict containing the configuration information or the filename
                of a file containing the configuration information.
        Raises:
            TypeError: Raises a TypeError if `config` is neither a dict nor a str.
        """
        if isinstance(config, str):
            self._config = Config.from_file(config)
        elif isinstance(config, dict):
            self._config = Config(config)
        else:
            raise TypeError("Invalid type for config, only dictionary or string is supported")
        
        for k,v in nec_args.items():
            if k not in self._config:
                self._config[k] = v
        
        if MEATCONFIG['half']:
            MEATCONFIG['dtype'] = torch.half
        else:
            MEATCONFIG['dtype'] = torch.float

MEATCONFIG = MetaConfig()

