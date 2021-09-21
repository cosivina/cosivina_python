from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

boostStimulusSpec = [
    ('amplitude', floatType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + boostStimulusSpec)
class BoostStimulus(Element):
    ''' Constant scalar stimulus. '''
    initElement = Element.__init__

    def __init__(self, label, amplitude = 0.):
        '''
        Args:
            label (str): Element label.
            amplitude (float): Value of the scalar stimulus.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'amplitude': PS_INIT_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.amplitude = amplitude

    def init(self):
        self.output = self.amplitude * np.ones((1, 1), dtype=floatType)

    def step(self, time, deltaT):
        pass
