#PlotBoxAndWhisker.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import matplotlib.pyplot as plt, numpy as np
from .plot_func import format_plot

def BoxplotPairPlotter_simple(ax,x1_values,x2_values,x1_label,x2_label,position1=0,position2=1,
                              xlabel='Num. Errors',
                              ylabel='',
                              fontsize=14,
                              vert=False,
                              notch=True,
                              widths=0.4,
                              manage_ticks=False,
                              **kwargs):
    """
    kwargs are passed to ax.boxplot directly
    kwargs are passed to format_plot directly

    Example Usage:
ax = BoxplotPairPlotter_simple(ax,x1_values,x2_values,x1_label,x2_label,position1=0,position2=1,
                              xlabel='Num. Errors',ylabel='',fontsize=14,
                              vert=False,notch=True,widths=0.4,manage_ticks=False)#,**kwargs)
    """
    #plot boxplots
    ax.boxplot(x1_values,vert=vert,notch=notch,widths=widths,manage_ticks=manage_ticks,positions=[position1],**kwargs)
    ax.boxplot(x2_values,vert=vert,notch=notch,widths=widths,manage_ticks=manage_ticks,positions=[position2],**kwargs)
    #format
    ax.set_yticks([position1,position2])
    ax.set_yticklabels([x1_label,x2_label])
    format_plot(xlabel=xlabel,ylabel=ylabel,fontsize=fontsize,**kwargs)
    return ax
