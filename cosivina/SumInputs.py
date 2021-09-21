from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

sumInputsSpec = [
    ('size', sizeTupleType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + sumInputsSpec)
class SumInputs(Element):
    ''' Computes the sum of several inputs of compatible sizes. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1)):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the output.
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
        self.output[:] = 0.
        for i in range(len(self.inputs)):
            self.output += self.inputs[i]
