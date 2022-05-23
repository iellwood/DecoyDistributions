import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'Arial', 'font.weight':'bold'})
fontdict_axis_label = {
    'fontfamily': 'Arial',
    'fontweight': 'bold',
    'fontsize': 11
}
fontdict_title = {
    'fontfamily': 'Arial',
    'fontweight': 'bold',
    'fontsize': 12
}


def x_axis_only(axis=None):
    if axis is None:
        axis = plt.gca()
    axis.get_yaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    # change all spines
    for which_axis in ['top', 'bottom', 'left', 'right']:
        axis.spines[which_axis].set_linewidth(2)
    axis.tick_params(width=2, length=5)

def no_box(axis=None):
    if axis is None:
        axis = plt.gca()
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # change all spines
    for which_axis in ['top', 'bottom', 'left', 'right']:
        axis.spines[which_axis].set_linewidth(2)
    axis.tick_params(width=2, length=5)


colors = {
    'green'  : [0.000, 0.639, 0.408],
    'blue'   : [0.000, 0.525, 0.749],
    'red'    : [0.769, 0.008, 0.137],
    'yellow' : [1.000, 0.827, 0.000],
    'purple' : [0.549, 0.000, 0.749],
    'orange' : [1.000, 0.502, 0.000],
    'pink'   : [0.945, 0.569, 0.608],
    'black'  : [0.000, 0.000, 0.000],
    'white'  : [1.000, 1.000, 1.000],
    'gray'   : [0.500, 0.500, 0.500],
}

def xlabel(label, axis=None):
    if axis is None:
        plt.xlabel(label, fontdict=fontdict_axis_label)
    else:
        axis.set_xlabel(label, fontdict=fontdict_axis_label)

def ylabel(label, axis=None):
    if axis is None:
        plt.ylabel(label, fontdict=fontdict_axis_label)
    else:
        axis.set_ylabel(label, fontdict=fontdict_axis_label)

def title(label, axis=None):
    if axis is None:
        plt.title(label, fontdict=fontdict_title)
    else:
        axis.set_title(label, fontdict=fontdict_title)


color_list = [colors['blue'], colors['red'], colors['green'], colors['orange'], colors['purple'], colors['yellow'], colors['pink'], colors['gray']]
