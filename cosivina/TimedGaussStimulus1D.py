from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

timedGaussStimulus1DSpec = [
    ('size', sizeTupleType),
    ('sigma', floatType),
    ('amplitude', floatType),
    ('position', floatType),
    ('circular', boolType),
    ('on_times', arrayType2D),
    ('normalized', boolType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + timedGaussStimulus1DSpec)
class TimedGaussStimulus1D(Element):
    ''' Creates a one-dimensional Gaussian stimulus active at specified time intervals. '''
    initElement = Element.__init__

    def __init__(self, label, size=(1, 1), sigma=1., amplitude=0., position=1., on_times=np.zeros((1,2)), circular=True, normalized=False):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the output.
            sigma (float): Width parameter of the Gaussian.
            amplitude (float): Amplitude of the Gaussian.
            position (float): Center of the Gaussian.
            on_times (numpy.ndarray): A 2D array of shape (n, 2),
                where `n` is the number of intervals at which
                the input is active.
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
            'on_times': PS_INIT_REQUIRED,
            'normalized': PS_INIT_REQUIRED
        })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigma = sigma
        self.amplitude = amplitude
        self.position = position
        self.on_times = on_times
        # Check that on_times is a two-column array
        if on_times.shape[1] != 2:
            raise ValueError("on_times must be an Nx2 matrix.")
        self.circular = circular
        self.normalized = normalized

        self.on = False
        self.stimulus_pattern = None

        # self.init()

    def init(self):
        self.output = np.zeros(self.size)
        if self.circular:
            self.stimulus_pattern = self.amplitude * circularGauss(np.arange(1, self.size[1] + 1),
                                                            self.position, self.sigma, self.normalized)
        else:
            self.stimulus_pattern = self.amplitude * gauss(np.arange(1, self.size[1] + 1),
                                                    self.position, self.sigma, self.normalized)


    def step(self, time, deltaT):
        should_be_on = np.any((time >= self.on_times[:, 0]) & (time <= self.on_times[:, 1]))
        if not self.on and should_be_on:
            self.output[0] = self.stimulus_pattern
            self.on = True
        elif self.on and not should_be_on:
            self.output.fill(0)
            self.on = False