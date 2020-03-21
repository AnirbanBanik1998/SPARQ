import matplotlib.pyplot as plt

def single_display(x, y, color, xlabel, ylabel):

    plt.plot(x, y, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()