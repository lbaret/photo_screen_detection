from os.path import join, abspath, dirname

class Path:
    CONFIG = abspath(dirname(__file__))
    MODULE = dirname(CONFIG)
    ROOT = dirname(MODULE)
    DATA = join(ROOT, 'data')
    NOTEBOOKS = join(ROOT, 'notebooks')