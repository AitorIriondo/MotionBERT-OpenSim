# opensim_shim.py - Shim module to make pyopensim compatible with opensim API
"""
This module creates compatibility between pyopensim and the expected opensim API.
pyopensim separates classes into submodules, while opensim exports everything at top level.
"""

import pyopensim

# Re-export everything from pyopensim
from pyopensim import *

# Import from submodules and expose at top level
from pyopensim.common import *
from pyopensim.simulation import *
from pyopensim.tools import *
from pyopensim.actuators import *
from pyopensim.analyses import *

# Make sure Logger is accessible
from pyopensim.common import Logger, LogSink, StringLogSink

# Expose version info
__version__ = pyopensim.__version__
