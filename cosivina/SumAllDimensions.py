from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

sumAllDimensionsSpec = [
    ('size', sizeTupleType),
    ('verticalSum', arrayType2D),
    ('horizontalSum', arrayType2D),
    ('fullSum', arrayType2D)
]

@jitclass(elementSpec + sumAllDimensionsSpec)
class SumAllDimensions(Element):
    ''' Computes horizontal, vertical, and total sum of 2D input.'''
    initElement = Element.__init__

    def __init__(self, label, inputSize = (1, 1)):
        '''
        Args:
            label (str): Element label.
            inputSize (tuple of int): Size of the input.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED
            })
        self.components = makeComponentList(['verticalSum', 'horizontalSum', 'fullSum'])
        self.defaultOutputComponent = 'fullSum'

        self.size = inputSize


    def init(self):
        self.verticalSum = np.zeros((1, self.size[1]))
        self.horizontalSum = np.zeros((1, self.size[0]))
        self.fullSum = np.zeros((1, 1))


    def step(self, time, deltaT):
        self.verticalSum[0] = np.sum(self.inputs[0], 0)
        self.horizontalSum[0] = np.sum(self.inputs[0], 1)
        self.fullSum[0][0] = np.sum(self.inputs[0])
