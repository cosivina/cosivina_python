from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

expandDimension2DSpec = [
    ('expandDimension', intType),
    ('size', sizeTupleType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + expandDimension2DSpec)
class ExpandDimension2D(Element):
    ''' Expand 1D input to 2D array along one dimension. '''
    initElement = Element.__init__

    def __init__(self, label, expandDimension = 1, outputSize = (1, 1)):
        '''
        Args:
            label (str): Element label.
            expandDimension (int): The dimension along which the input
                is expanded (1 for vertical, 2 for horizontal).
            outputSize (tuple of int): Size of the resulting output.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'expandDimension': PS_FIXED,
                'size': PS_FIXED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        if not (expandDimension == 1 or expandDimension == 2):
            raise ValueError('Argument expandDimensions must be either 1 or 2.')
        self.expandDimension = expandDimension
        self.size = outputSize


    def init(self):
        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        if self.expandDimension == 1:
            self.output[:][:] = np.reshape(self.inputs[0], (intType(1), self.size[1]))
        else:
            self.output[:][:] = np.reshape(self.inputs[0], (self.size[0], intType(1)))

