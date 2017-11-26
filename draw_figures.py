"""
define class for drawing curves
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.edgecolor': '0.0',
                            'grid.color': '.8', 'legend.frameon': True})
# if os.environ.get('DISPLAY', '') == '':
#     plt.switch_backend('agg')


class Line(object):
    """
    single line to be plot
    """

    def __init__(self, label, data, line_style="-", color="blue", line_width=3):
        self.label = label
        self.data = data
        self.line_style = line_style
        self.color = color
        self.line_width = line_width


class Figure(object):
    """
    figure
    """

    def __init__(self, figure_name, save_path="./",
                 xlabel=None, ylabel=None,
                 title=None, x_lim=None, y_lim=None):
        self.figure_name = figure_name
        self.save_path = save_path
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.x_lim = x_lim
        self.y_lim = y_lim

    def draw(self, lines):
        """
        draw figure
        """
        if not isinstance(lines, list):
            lines = [lines]

        plt.figure()
        for items in lines:
            data_len = len(items.data)
            x = np.linspace(1, data_len, data_len)
            plt.plot(x, items.data, items.line_style,
                     linewidth=items.line_width,
                     color=items.color,
                     label=items.label)

        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        if self.x_lim is not None:
            plt.xlim(self.x_lim)
        if self.y_lim is not None:
            plt.ylim(self.y_lim)
        if self.title is not None:
            plt.title(self.title)
        plt.legend(loc=0, fontsize="small")

        plt.savefig(os.path.join(self.save_path,
                                 self.figure_name + ".pdf"), format="pdf")
        plt.savefig(os.path.join(self.save_path,
                                 self.figure_name + ".png"), format="png")


class TextParse(object):
    """
    parse text
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def __call__(self):
        with open(self.file_path, "r") as f:
            for info in f.readlines():
                info_split = info.split('\t')
                temp_data = np.zeros(len(info_split) - 1)
                for i in range(len(info_split) - 1):
                    temp_data[i] = float(info_split[i])
                if self.data is None:
                    self.data = temp_data
                else:
                    self.data = np.vstack((self.data, temp_data))
        self.data = np.transpose(self.data)
        return self.data


def main():
    data_base = []

    file_parse = TextParse("./database/cifar10_resnet44_baseline.txt")
    data_base.append(file_parse())

    file_parse = TextParse("./database/cifar10_resnet56_baseline.txt")
    data_base.append(file_parse())

    # -------------------------------------------------------
    # draw testing curves
    lines = []
    lines.append(Line(label="ResNet-44",
                      data=data_base[0][3], color="blue"))

    lines.append(Line(label="ResNet-56",
                      data=data_base[1][3], color="red"))

    figure = Figure(figure_name="cifar10_resnet_testing",
                    x_lim=[150, 400], y_lim=[5, 10],
                    xlabel="Epoch",
                    ylabel="Testing error (%)")

    figure.draw(lines=lines)

    # -------------------------------------------------------
    # draw training curves
    lines = []
    lines.append(Line(label="ResNet-44",
                      data=data_base[0][1], color="blue"))

    lines.append(Line(label="ResNet-56",
                      data=data_base[1][1], color="red"))

    figure = Figure(figure_name="cifar10_resnet_training",
                    y_lim=[0, 15],
                    xlabel="Epoch",
                    ylabel="Training error (%)")

    figure.draw(lines=lines)


if __name__ == '__main__':
    main()
