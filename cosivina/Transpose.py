from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

transposeSpec = [
    ('size', sizeTupleType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + transposeSpec)
class Transpose(Element):
    ''' Transposes its input. '''
    initElement = Element.__init__

    def __init__(self, label, outputSize = (1, 1)):
        '''
        Args:
            label (str): Element label.
            outputSize (tuple of int): Size of the transposition result.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = outputSize


    def init(self):
        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        self.output[:][:] = np.transpose(self.inputs[0])


