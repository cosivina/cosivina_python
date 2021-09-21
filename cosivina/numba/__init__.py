import cosivina.options as options
options.useNumba = True

from cosivina.Simulator import Simulator
from cosivina.elements import *

from cosivina.auxiliary import getNumbaStatus
options.useNumba = getNumbaStatus()