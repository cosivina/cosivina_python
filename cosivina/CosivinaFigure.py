import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


class CosivinaFigure:
    ''' Generate an animation of specified components from the simulator. '''
    def __init__(self, simulator, gridsize, figsize=(10,6)):
        '''
        simulator (Cosivina.Simulator): A simulator instance providing data for plots.
        gridsize (tuple of int): Tuple (nrows, ncols) specifying grid layout.
        figsize (tuple of int): Tuple (width, height) specifying the size of the figure.
        '''
        self.simulator = simulator
        self.gridsize = gridsize
        self.figsize = figsize
        self.fig = plt.figure(figsize=figsize)  # initialize a figure
        self.grid_spec = GridSpec(gridsize[0], gridsize[1], figure=self.fig)
        self.axes = {}  # dictionary containing all axes with a key in the format of (rowpos, colpos)
        self.axes_range = {}  # dictionary keeping axes ranges
        self.frames = {}  # dictionary of lists with all the frames for animation
        self.frame_num = 0  # count the total number of frames aggregated over all plots

    def addGrid(self, pos, label, activation_range=(-5.5, 5.5), **kwargs):
        """
        Adds an axis to the grid at a specified location and size.
        pos (tuple of int): tuple (row, col, rowspan, colspan) specifying the location and span.
        activation_range (tuple of float): set the activation range to plot which are depicted differently
        depending on the figure type.
        """
        ax = self.fig.add_subplot(self.grid_spec[pos[0]:pos[0] + pos[2], pos[1]:pos[1] + pos[3]])
        self.axes[(pos[0], pos[1])] = ax
        self.axes_range[(pos[0], pos[1])] = activation_range
        self.frames[(pos[0], pos[1])] = []  # List to hold frame data for multiple plots
        # axis settings
        ax.set_title(label)
        ax.set_xlabel(kwargs.get('xlabel'), loc='right', labelpad=-5)
        ax.set_ylabel(kwargs.get('ylabel'), loc='top', labelpad=-30, rotation=0)
        ax.set_xticks(kwargs.get('xticks'), labels=kwargs.get('xticklabels')) if kwargs.get('xticks') is not None else None
        ax.set_yticks(kwargs.get('yticks'), labels=kwargs.get('yticklabels')) if kwargs.get('yticks') is not None else None

    def addPlot(self, element, figtype, loc, component="output", manual_data=None,  **kwargs):
        """
        Adds a plot to the specified axis location with data from simulator.
        manual_data (numpy.ndarray): use a custom data array if it is provided instead of element and component labels.
        """
        if manual_data is None:
            plot_data = self.simulator.getComponent(element, component)
        else:
            plot_data = manual_data
        ax = self.axes[loc]

        # Handle different figtypes and store data appropriately
        if figtype == "horizontal":
            line, = ax.plot(plot_data[0], c=kwargs.get('c'), label=element.replace("Hist", ""))
            ax.set_ylim(self.axes_range[loc])
        elif figtype == 'vertical':
            line, = ax.plot(plot_data[0], np.arange(len(plot_data[0])), c=kwargs.get('c'), label=element.replace("Hist", ""))
            ax.set_xlim(self.axes_range[loc])
        elif figtype == "2D":
            line = ax.imshow(plot_data[0], interpolation='bilinear', origin='lower',
                             cmap='jet', aspect='auto', vmin=self.axes_range[loc][0], vmax=self.axes_range[loc][1])
        elif figtype == "dot":
            x_data = kwargs.get('x', np.arange(len(plot_data[0])))  # at which x value the node will be plotted
            line = ax.scatter(x=x_data, y=plot_data[0], c=kwargs.get('c'), label=element.replace("Hist", ""))
            ax.set_ylim(self.axes_range[loc])
            ax.axes.get_xaxis().set_visible(False)
            plot_data = (x_data, plot_data)  # separate data for each plot within an axis
        else:
            raise TypeError("figtype not defined")

        # Append plot details for multi-plot updates
        self.frames[loc].append((line, figtype, plot_data))
        self.frame_num = len(plot_data) if self.frame_num == 0 else self.frame_num
        self.fig.tight_layout()

    def getAnim(self, frame_step_interval=1):
        """
        Creates and displays the animation.
        frame_step_interval (int): Animate only every n-th frame.
        return (matplotlib.animation.FuncAnimation)
        """
        matplotlib.rcParams['animation.embed_limit'] = 40  # in MB
        self.animation = FuncAnimation(self.fig, self._run,
                                       frames=range(0, self.frame_num, frame_step_interval),
                                       repeat=False)
        return self.animation

    def _run(self, frame_number):
        """
        Updates each frame for the animation.
        """
        for loc, ax_plots in self.frames.items():
            for line, figtype, plot_data in ax_plots:
                if figtype == "horizontal":
                    line.set_ydata(plot_data[frame_number])
                elif figtype == 'vertical':
                    line.set_xdata(plot_data[frame_number])
                elif figtype == "2D":
                    line.set_data(plot_data[frame_number])
                elif figtype == "dot":
                    x_data, y_data = plot_data
                    offsets = np.column_stack((x_data, y_data[frame_number]))
                    line.set_offsets(offsets)
