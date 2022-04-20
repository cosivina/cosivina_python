from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

pointwiseProductSpec = [
    ('size', sizeTupleType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + pointwiseProductSpec)
class PointwiseProduct(Element):
    ''' Multiplies corresponding entries from two input components. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1)):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of output (sizes of input 
                components must be compatible with output size).
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'
        self.size = size

    def init(self):
        self.output = np.zeros(self.size)

    def step(self, time, deltaT):
        self.output[:][:] = self.inputs[0] * self.inputs[1]
