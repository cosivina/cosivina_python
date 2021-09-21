from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

gaussStimulus1DSpec = [
    ('size', sizeTupleType),
    ('sigma', floatType),
    ('amplitude', floatType),
    ('position', floatType),
    ('circular', boolType),
    ('normalized', boolType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + gaussStimulus1DSpec)
class GaussStimulus1D(Element):
    ''' One-dimensional Gaussian stimulus. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigma = 1., amplitude = 0.,
            position = 1., circular = True, normalized = False):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the output.
            sigma (float): Width parameter of the Gaussian.
            amplitude (float): Amplitude of the Gaussian.
            position (float): Center of the Gaussian.
            circular (bool): Flag indicating whether Gaussian is
                defined over circular space.
            normalized (bool): Flag indicating whether Gaussian is
                normalized before scaling with amplitude.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'sigma': PS_INIT_REQUIRED,
                'amplitude': PS_INIT_REQUIRED,
                'position': PS_INIT_REQUIRED,
                'circular': PS_INIT_REQUIRED,
                'normalized': PS_INIT_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigma = sigma
        self.amplitude = amplitude
        self.position = position
        self.circular = circular
        self.normalized = normalized

    def init(self):
        self.output = np.zeros(self.size)
        if self.circular:
            self.output[0] = self.amplitude * circularGauss(np.arange(1, self.size[1] + 1),
                    self.position, self.sigma, self.normalized)
        else:
            self.output[0] = self.amplitude * gauss(np.arange(1, self.size[1] + 1),
                    self.position, self.sigma, self.normalized)


    def step(self, time, deltaT):
        pass
