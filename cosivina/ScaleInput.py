from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

scaleInputSpec = [
    ('size', sizeTupleType),
    ('amplitude', floatType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + scaleInputSpec)
class ScaleInput(Element):
    ''' Scales its input with a constant factor. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), amplitude=0.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of input and output.
            amplitude (float): Scaling factor.
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
        self.output[:][:] = self.amplitude * self.inputs[0]
