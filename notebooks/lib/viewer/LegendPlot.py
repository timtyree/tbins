#LegendPlot.py plots legends separately
#Programmer: Tim Tyree
#Date: 6.7.2022
import matplotlib as mpl
import matplotlib.pyplot as plt

def PlotLegend(label_lst=["Concept Cells", "Remaining"],color_lst=['C1', 'C0'],figsize=(1,1),use_show=True,**kwargs):
    """PlotLegend plots a simple legend as defined by label_lst,color_lst.
    kwargs is passed to matplotlib.pyplot.legend

    Example Usage: make a legend that says "Concept Cells" and "Remaining"
label_lst=["Concept Cells", "Remaining"]
color_lst=['C1', 'C0']
kwargs_legend={
    'frameon':True,
    'loc':'center',
    'ncol':1,
    'fontsize':18}
PlotLegend(label_lst=label_lst,color_lst=color_lst,figsize=(1,1),**kwargs_legend);
    """
    fig,ax=plt.subplots(figsize=figsize)
    #create a color palette
    palette = dict(zip(label_lst, color_lst))
    #create legend handles manually
    handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
    #create legend
    plt.legend(handles=handles,**kwargs)
    #get current axes object and turn off axis
    plt.gca().set_axis_off()
    plt.tight_layout()
    if use_show:
        plt.show()
    return True

def PlotLegend_lines(label_lst=["Random Guess", "5X Random Guess"],color_lst=['r', 'k'],
                     linestyle='dashed',
                     lw=2,
                     figsize=(1,1),use_show=True,**kwargs):
    """PlotLegend_lines plots a simple legend as defined by label_lst,color_lst.
    kwargs is passed to matplotlib.pyplot.legend

    Example Usage:
label_lst=["Random Guess", "5X Random Guess"]
color_lst=['r', 'k']
kwargs_legend={
    'frameon':False,
    'loc':'center',
    'ncol':2,
    'fontsize':14}
PlotLegend_lines(label_lst=label_lst,color_lst=color_lst,figsize=(1,1),**kwargs_legend);
    """
    fig,ax=plt.subplots(figsize=figsize)
    #create a color palette
    palette = dict(zip(label_lst, color_lst))
    #create legend handles manually
    handles = [mpl.patches.mlines.Line2D([],[],color=palette[x], label=x, linestyle=linestyle,lw=lw) for x in palette.keys()]
    #create legend
    plt.legend(handles=handles,**kwargs)
    #get current axes object and turn off axis
    plt.gca().set_axis_off()
    plt.tight_layout()
    if use_show:
        plt.show()
    return True
