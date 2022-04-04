"""Top-level package for BRUNO."""
from types import ModuleType
from importlib import import_module

__author__ = """Amrit Singh"""
__email__ = 'amrit.singh@hli.ubc.ca'
__version__ = '0.1.0'

class LazyLoader(ModuleType):
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self):
        module = import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)

data = LazyLoader('data', globals(), 'bruno.data')
nn = LazyLoader('nn', globals(), 'bruno.nn')
learn = LazyLoader('learn', globals(), 'bruno.learn')

__version__ = '0.1.0'