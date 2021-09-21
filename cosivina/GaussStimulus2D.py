from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

gaussStimulus2DSpec = [
    ('size', sizeTupleType),
    ('sigmaY', floatType),
    ('sigmaX', floatType),
    ('amplitude', floatType),
    ('positionY', floatType),
    ('positionX', floatType),
    ('circularY', boolType),
    ('circularX', boolType),
    ('normalized', boolType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + gaussStimulus2DSpec)
class GaussStimulus2D(Element):
    ''' Two-dimensional Gaussian stimulus. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigmaY = 1., sigmaX = 1.,
            amplitude = 0., positionY = 1., positionX = 1.,
            circularY = True, circularX = True, normalized = False):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the output.
            sigmaY (float): Vertical width parameter.
            sigmaX (float): Horizontal width parameter.
            amplitude (float): Amplitude of the Gaussian.
            positionY (float): Vertical center of the Gaussian.
            positionX (float): Horizontal center of the Gaussian.
            circularY (bool): Flag indicating whether Gaussian is
                defined over circular space in vertical dimension.
            circularX (bool): Same for horizontal dimension.
            normalized (bool): Flag indicating whether Gaussian is
                normalized before scaling with amplitude.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'sigmaY': PS_INIT_REQUIRED,
                'sigmaX': PS_INIT_REQUIRED,
                'amplitude': PS_INIT_REQUIRED,
                'positionY': PS_INIT_REQUIRED,
                'positionX': PS_INIT_REQUIRED,
                'circularY': PS_INIT_REQUIRED,
                'circularX': PS_INIT_REQUIRED,
                'normalized': PS_INIT_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigmaY = sigmaY
        self.sigmaX = sigmaX
        self.amplitude = amplitude
        self.positionY = positionY
        self.positionX = positionX
        self.circularY = circularY
        self.circularX = circularX
        self.normalized = normalized

    def init(self):
        self.output = self.amplitude * circularGauss2d(
                np.arange(1, self.size[0] + 1), np.arange(1, self.size[1] + 1),
                self.positionY, self.positionX, self.sigmaY, self.sigmaX,
                self.circularY, self.circularX, self.normalized)

    def step(self, time, deltaT):
        pass
