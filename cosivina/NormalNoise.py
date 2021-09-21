from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

normalNoiseSpec = [
    ('size', sizeTupleType),
    ('amplitude', floatType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + normalNoiseSpec)
class NormalNoise(Element):
    ''' Normally distributed noise.

    Creates an array of independent normally distributed random values
    in each step. Note: The strength of the noise is scaled with
    1/sqrt(deltaT) so that it will be effectively be scaled with
    sqrt(deltaT) when used as input in the field equation. Note that
    the scaling with 1/tau is also applied to all inputs in the field
    equation, and may have to be compensated for in the noise amplitude.
    '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), amplitude = 0.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the output array.
            amplitude (float): Factor with which random values (from
                standard normal distribution) are scaled.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'amplitude': PS_CHANGEABLE
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.amplitude = amplitude

    def init(self):
        self.output = np.zeros(self.size)

    def step(self, time, deltaT):
        self.output[:][:] = 1. / np.sqrt(deltaT) * \
                np.random.normal(0, self.amplitude, self.size)
