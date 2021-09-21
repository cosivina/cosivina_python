from cosivina.base import *

elementSpec = []

class Element(object):
    ''' Base class for elements. '''
    def __init__(self, label):
        self.label = label
        self.parameters = {}
        self.components = {}
        self.defaultOutputComponent = ''
        self.inputElementLabels = []
        self.inputComponents = []
        self.inputs = []
        self.nInputs = 0

    def init(self):
        ''' Initialize element. '''
        pass

    def step(self, time, deltaT):
        ''' Perform a single simulation step. '''
        pass

    def close(self):
        ''' Close any connections to files or hardware. '''
        pass

    def isParameter(self, name):
        ''' Check if string is name of parameter in element. '''
        return name in self.parameters

    def isComponent(self, name):
        ''' Check if string is name of component in element. '''
        return name in self.components

    def parameterList(self):
        ''' Get a list of element's parameter names. '''
        return list(self.parameters.keys())

    def getParameterStatus(self, name):
        ''' Get int value indicating change status of parameter. '''
        if not name in self.parameters:
            raise ValueError('Parameter name not found.')
        return self.parameters[name]

    def addInput(self, inputElementLabel, inputComponent):
        ''' Add an input to the element. '''
        self.inputElementLabels.append(inputElementLabel)
        self.inputComponents.append(inputComponent)
        self.nInputs += 1



if options.useNumba:
    elementSpec = [
        ('label', stringType),
        ('parameters', types.DictType(stringType, intType)),
        ('components', types.DictType(stringType, intType)),
        ('defaultOutputComponent', stringType),
        ('inputElementLabels', stringListType),
        ('inputComponents', stringListType),
        ('inputs', types.ListType(arrayType2D)),
        ('nInputs', intType)
    ]

    def numbaInit(self, label):
        self.label = label
        self.parameters = typed.Dict.empty(stringType, intType)
        self.components = typed.Dict.empty(stringType, intType)
        self.defaultOutputComponent = ''
        self.inputElementLabels = typed.List.empty_list(stringType)
        self.inputComponents = typed.List.empty_list(stringType)
        self.inputs = typed.List.empty_list(arrayType2D)
        self.nInputs = intType(0)

    Element.__init__ = numbaInit


