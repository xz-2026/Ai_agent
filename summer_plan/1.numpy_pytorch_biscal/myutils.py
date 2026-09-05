import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import cycle
#使用svg格式在jupyter中显示绘图
def use_svg_display(): 

    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.rcParams['svg.fonttype'] = 'none'
"""此函数设置坐标系。 xscale/yscale控制坐标轴的刻度映射方法,默认linear线性,最常用的还有log对数坐标。刻数直接按10,10^2,10^3等走"""
def set_axes(axes,xlabel,ylabel,xlim=None,ylim=None,xscale='linear',yscale='linear',legend=None):
    # 设置matplotlib在jupyter里默认输出svg矢量图（更清晰）
    axes.set(xlabel=xlabel,ylabel=ylabel,xscale=xscale,yscale=yscale,xlim=xlim,ylim=ylim)
    if legend is not None and len(legend)>0:
        axes.legend(legend)
    axes.grid()
    
def plot_func(X,Y=None,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',
              fmts=('-','m--','g-','r:'),figsize=(3.5,2.5),axes=None):
    if legend is None:
        legend=[]
    if axes is None:
        _,axes=plt.subplots(figsize=figsize)
    if Y is None:
        X,Y=range(len(X)),X

    if not isinstance(Y,(list,tuple)):
        Y=[Y]
    # X对齐Y长度
    if not isinstance(X,(list,tuple)) or len(X)!=len(Y):
        X = [X]*len(Y)

    # cycle 无限循环线条样式，不怕曲线比fmts多
    for x,y,fmt in zip(X,Y,cycle(fmts)):
        # 关键：torch张量转numpy，去掉多余维度
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().squeeze()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().squeeze()
        axes.plot(x,y,fmt)
    set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend)