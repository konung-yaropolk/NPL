# %%
#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
import seaborn as sns
import pandas as pd

# Only use for plot layout adjustment
DEBUG = False
#%%
class BarGraph:
    '''
        Input format:

        data: array of arrays, where each array is a feature. So the length of array of arrays is a number of features.

    '''
    def __init__(self,
                 data:list,
                 mask = None) -> None:

        self.raw_data = data
        self.length = len(data)
        # self.default_bar_colors
        # self.default_border_colors
        
        
        
        
    #params how wide is the bar
    #the number of bars -- the number of lists
    #params if you want filling
    #params the edges of bar the color and thickness
    #if you want stds or not
    #width is from 0 and 1 as well as the coordinates, the bars can overlap
    #option for a log scale
    #also add horizontal barplot
    
    #vertical means if bars are horizontal or vertical
    #barwidth width of a bar from 0 to 1
    
    #groupBy if you want certain bars to be grouped within each for example if you 
    #have the same experiment with differenet groups
    #also add legend
    
    
    
    ### vertical : draws horizontal or vertical barplot, default is vertical ###
    ### barWidth : regulates bar thickness, varies from 0.0 to 1, where 1 yields no space in between bars, default is 0.9 ###
    ### colorsBarsFill : 
        
    def draw(self,
             vertical = True,
             barWidth = 0.9,
             colorsBarsFill = None,
             colorsBarsBorder = None,
             noBorder = False,
             noFill = True,
             errBar = True,
             errBarCapWidth = 0.0,
             scatterData = True,
             groupBy = None, #should be a number
             groupByAxis = None,
             tickLabels = [0,0,0], 
             markers = None,
             annotate = True,
             annotateLabels = None):
        
        
        #### Make data ###
        y = self.raw_data
        x = list(range(len(y))) # x-coordinates of your bars
        
        
        
        ### Take into account all the parameters ###
        
        ##we need to check if this is a singular value if it is color like
        ##if there are multiple colors we need to make sure that they are the same length as data
        ##and also that they are 
        if (colorsBarsFill==None
           or (not hasattr(colorsBarsFill, "__len__") and color.is_color_like(colorsBarsFill)==False) #check for a singular value
           or (len(colorsBarsFill)!=self.length) #if the number of colors dont match the number of bars
           or (not all(color.is_color_like(i) for i in colorsBarsFill))): #if the numbers are not correctly formatted
            colorsBarsFill = plt.cm.Set3(np.arange(len(self.raw_data)))
        
    
            
        if colorsBarsBorder==None:
            colorsBarsBorder = plt.cm.Set3(np.arange(len(self.raw_data)))
            
            
        # if groupBy: call the function itself
        # # markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
        # # colors = ['r','g','b','c','m', 'y', 'k']
        # #also add subplots
        fig, ax = plt.subplots()
        #compute error Bars
        if errBar:
            stds = [np.std(yi) for yi in y]
        #compute bar heights
        bar_heights = [np.mean(yi) for yi in y]
        
        
        ### Draw a vertical main plot ###
        if vertical:
            ax.bar(x,
                height=bar_heights,
                yerr=stds,    # error bars
                # ecolor = colors,
                capsize=12, # error bar cap width in points
                width=barWidth,    # bar width
                # tick_label=["control", "test"],
                color=colorsBarsFill,  # face color transparent
                edgecolor=colorsBarsBorder,
                # linewidth = 4
                #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
            )
        ### Draw a horizonatal main plot ###
        else:
            print(x,y)
            ax.barh(x,
                bar_heights,
                height=barWidth,
                yerr=stds,    # error bars
                # # ecolor = colors,
                # capsize=12, # error bar cap width in points
                # width=barWidth,    # bar width
                # # tick_label=["control", "test"],
                # color=(0,0,0,0),  # face color transparent
                # # edgecolor=colors,
                # linewidth = 4
                #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
            )
    #     for i in range(len(x)):
    #         # distribute scatter randomly across whole width of bar
    #         ax.scatter(x[i] + np.random.random(y[i].size) * barWidth - barWidth / 2, y[i], marker = 'v',color=colors[i])

    #     plt.show()
    # def compose():
        
#     def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
#     """ 
#     Annotate barplot with p-values.

#     :param num1: number of left bar to put bracket over
#     :param num2: number of right bar to put bracket over
#     :param data: string to write or number for generating asterixes
#     :param center: centers of all bars (like plt.bar() input)
#     :param height: heights of all bars (like plt.bar() input)
#     :param yerr: yerrs of all bars (like plt.bar() input)
#     :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
#     :param barh: bar height in axes coordinates (0 to 1)
#     :param fs: font size
#     :param maxasterix: maximum number of asterixes to write (for very small p-values)
#     """

#     if type(data) is str:
#         text = data
#     else:
#         # * is p < 0.05
#         # ** is p < 0.005
#         # *** is p < 0.0005
#         # etc.
#         text = ''
#         p = .05

#         while data < p:
#             text += '*'
#             p /= 10.

#             if maxasterix and len(text) == maxasterix:
#                 break

#         if len(text) == 0:
#             text = 'n. s.'

#     lx, ly = center[num1], height[num1]
#     rx, ry = center[num2], height[num2]

#     if yerr:
#         ly += yerr[num1]
#         ry += yerr[num2]

#     ax_y0, ax_y1 = plt.gca().get_ylim()
#     dh *= (ax_y1 - ax_y0)
#     barh *= (ax_y1 - ax_y0)

#     y = max(ly, ry) + dh

#     barx = [lx, lx, rx, rx]
#     bary = [y, y+barh, y+barh, y]
#     mid = ((lx+rx)/2, y+barh)

#     plt.plot(barx, bary, c='black')

#     kwargs = dict(ha='center', va='bottom')
#     if fs is not None:
#         kwargs['fontsize'] = fs

#     plt.text(*mid, text, **kwargs)
# plt.figure()
# plt.bar(bars, heights, align='center')
# plt.ylim(0, 5)
# barplot_annotate_brackets(0, 1, .1, bars, heights)
# barplot_annotate_brackets(1, 2, .001, bars, heights)
# barplot_annotate_brackets(0, 2, 'p < 0.0075', bars, heights, dh=.2)

data = np.random.normal(100, 20, 200)
data = y = [np.random.random(30) * 2 + 5, np.random.random(10) * 3 + 8, np.random.random(10) * 3 + 2]
bar = BarGraph(data)
# Creating dataset
bar.draw(vertical = True, barWidth = 0.9, )


# use a decorator function to put what you want on the top of the parplots



# %%
# print(color.is_color_like('red'))
# print(color.is_color_like([0,0,0]))
# print(color.is_color_like([0,0]))


#bar.draw(vertical = True, barWidth = 0.9, colorsBarsFill = 'red')


# %%
