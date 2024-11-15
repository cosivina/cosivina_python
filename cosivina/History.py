from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

historySpec = [
    ("size", sizeTupleType),
    ("storingTimes", arrayType1D),
    ("output", arrayType2D),
]

@jitclass(elementSpec + historySpec)
class History(Element):
    '''Element that stores its input at specified times. A vector of
    simulation times [t_1, ..., t_K] must be specified.
    The input to the element at those times is then stored in
    a K x (N x M x ...) matrix. '''

    initElement = Element.__init__

    def __init__(self, label, size=(1, 1), storingTimes=[]):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the input.
            storingTimes (list of float): Vector of time steps (in int) at which the input is stored.
                Currently not implemented and defaults to every time steps
        '''
        self.initElement(label)
        self.parameters = makeParamDict({"size": PS_FIXED, "storingTimes": PS_FIXED})
        self.components = makeComponentList(["output"])
        self.defaultOutputComponent = "output"

        self.size = size
        self.storingTimes = storingTimes
        self.dimensionality = 0

    def init(self):
        if self.size == [1, 1]:  # node
            self.dimensionality = 0
        elif self.size[0] == 1:  # 1D
            self.dimensionality = 1
        elif len(self.size) == 2:
            self.dimensionality = 2  # 2D arrays
        else:
            raise TypeError("The class History does currently not support inputs of more than two dimensions")

        if self.dimensionality == 0:
            self.output = np.zeros((len(self.storingTimes) + 1))
        if self.dimensionality == 1:
            self.output = np.zeros((len(self.storingTimes) + 1, self.size[1]))
        else:
            self.output = np.zeros((len(self.storingTimes) + 1, self.size[0], self.size[1]))


    def step(self, time, deltaT):
        # TODO: history should receive inputs from the field before the activation is updated
            self.output[int(time / deltaT)] = self.inputs[0]
        # # TODO: storing history only at specified storingTimes
        # if np.any(self.storingTimes == time):  # store only at time specified by storingTimes
        #     self.output[np.nonzero(self.storingTimes == time)[0][0]] = self.inputs[0]
