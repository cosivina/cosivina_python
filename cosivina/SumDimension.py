from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

sumDimensionSpec = [
    ('sumDimensions', intArrayType),
    ('size', sizeTupleType),
    ('amplitude', floatType),
    ('dimensionOrder', intArrayType),
    ('output', arrayType2D)
]

@jitclass(elementSpec + sumDimensionSpec)
class SumDimension(Element):
    ''' Computes sum over one or dimensions of the input.

    Optionally, the result can be scaled with a fixed amplitude, and
    the shape of the formed sum can be changed.
    '''
    initElement = Element.__init__

    def __init__(self, label, sumDimensions = np.array([1]),
            outputSize = (1, 1), amplitude = 1.,
            dimensionOrder = np.array([1, 2])):
        '''
        Args:
            label (str): Element label.
            sumDimensions (numpy ndarray 1D): Dimension(s) of the input
                over which the sum is computed.
            outputSize (tuple of int): Size of the resulting output
                (may be used to change shape of the sum).
            amplitude (float): Scaling factor.
            dimensionOrder (numpy ndarray 1D): Currently not used (the
                shape of the output can be determined via outputSize).

        Note: Arguments sumDimensions and dimensionOrder must be
        one-dimensional numpy ndarrays. This may change in future
        versions.

        Examples:
        SumDimensions('vertical sum', np.ndarray([1]), (1, 100))
        SumDimensions('full sum', np.ndarray([1, 2]), (1, 1))
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'sumDimensions': PS_FIXED,
                'size': PS_FIXED,
                'amplitude': PS_CHANGEABLE,
                'dimensionOrder': PS_FIXED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        if sumDimensions.size == 0:
            raise TypeError('Argument "sumDimensions" must not be empty')
        self.sumDimensions = sumDimensions
        self.size = outputSize
        self.amplitude = amplitude
        self.dimensionOrder = dimensionOrder # not used, just kept for consistency with Matlab


    def init(self):
        if not (np.all(self.sumDimensions == np.array([1], dtype=intType))
                or np.all(self.sumDimensions == np.array([2], dtype=intType))
                or np.all(self.sumDimensions == np.array([1, 2], dtype=intType))
                or np.all(self.sumDimensions == np.array([2, 1], dtype=intType))):
            raise TypeError('Invalid parameter value for "sumDimensions".')
        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        if self.sumDimensions.size == 1 and self.sumDimensions[0] == 1:
            self.output[:][:] = self.amplitude * \
                    np.reshape(np.sum(self.inputs[0], 0), self.size)
        elif self.sumDimensions.size == 1 and self.sumDimensions[0] == 2:
            self.output[:][:] = self.amplitude * \
                    np.reshape(np.sum(self.inputs[0], 1), self.size)
        else:
            self.output[0][0] = self.amplitude * np.sum(self.inputs[0])

