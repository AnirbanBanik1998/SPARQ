import matplotlib.pyplot as plt

def single_display(x, y, color, xlabel, ylabel):

    plt.plot(x, y, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def double_display(x, y1, color1, label1, y2, color2, label2, 
        xlabel, ylabel):

    plt.plot(x, y1, color1, linewidth=2, label=label1)
    plt.plot(x, y2, color2, linewidth=2, label=label2)
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def quad_display(x, y1, color1, label1, y2, color2, label2, y3, color3, 
        label3, y4, color4, label4, xlabel, ylabel):
    
    plt.plot(x, y1, color1, linewidth=2, label=label1)
    plt.plot(x, y2, color2, linewidth=2, label=label2)
    plt.plot(x, y3, color3, linewidth=2, label=label3)
    plt.plot(x, y4, color4, linewidth=2, label=label4)
    plt.legend(loc='best', prop={'size': 30})
    plt.xlabel(xlabel, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.show()