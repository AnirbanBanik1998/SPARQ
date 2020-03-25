import matplotlib.pyplot as plt

def single_display(x, y, color, xlabel, ylabel):

    plt.plot(x, y, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def double_display(x1, y1, color1, label1, x2, y2, color2, label2,
        xlabel, ylabel):

    plt.plot(x1,y1,color1,label=label1)
    plt.plot(x2, y2, color2, label=label2)
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()