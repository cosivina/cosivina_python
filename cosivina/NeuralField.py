from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

neuralFieldSpec = [
    ('size', sizeTupleType),
    ('tau', floatType),
    ('h', floatType),
    ('beta', floatType),
    ('activation', arrayType2D),
    ('inputSum', arrayType2D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + neuralFieldSpec)
class NeuralField(Element):
    ''' Dynamic neural field.

    A dynamic neural field or set of discrete dynamic nodes of any
    dimensionality with sigmoid (logistic) output function. The
    field activation is updated according to the Amari equation.
    '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), tau = 10., h = -5., beta = 4.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Field size.
            tau (float): Time constant.
            h (float): Resting level.
            beta (float): Steepness of sigmoid output function.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'h': PS_CHANGEABLE,
                'tau': PS_CHANGEABLE,
                'beta': PS_CHANGEABLE
            })
        self.components = makeComponentList(['activation', 'output', 'inputSum'])
        self.defaultOutputComponent = 'output'
        self.size = size
        self.tau = tau
        self.h = h
        self.beta = beta

    def init(self):
        self.activation = np.zeros(self.size) + self.h
        self.output = sigmoid(self.activation, self.beta)
        self.inputSum = np.zeros(self.size)

    def step(self, time, deltaT):
        self.inputSum[:][:] = 0.
        for i in range(len(self.inputs)):
            self.inputSum += self.inputs[i] # getattr(self.inputs[i], self.inputComponents[i])
        self.activation += deltaT/self.tau * (- self.activation + self.h + self.inputSum)
        self.output[:] = sigmoid(self.activation, self.beta)
