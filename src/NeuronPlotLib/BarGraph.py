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
    
    #add note about modifying opacity
    #create a tool for python that allows choosing colors an color pallettes
    
    ### vertical : draws horizontal or vertical barplot, default is vertical ###
    ### barWidth : regulates bar thickness, varies from 0.0 to 1, where 1 yields no space in between bars, default is 0.9 ###
    ### colorsBarsFill : 
    
    
    
    def _identify_colors(self,
                         colorInput):
           ##we need to check if this is a singular value if it is color like
        ##if there are multiple colors we need to make sure that they are the same length as data
        ##and also that they are 
        if colorInput==None: 
            return None
        #check if it is one color using build in function
        if color.is_color_like(colorInput)==True:
            return "one color"
        #check if it is a valid array of colors and that is has an appropriate length
        elif (hasattr(colorInput, "__len__") and
              len(colorInput)==self.length and
              all(color.is_color_like(i) for i in colorInput)):
            return "color array"
        else:
            return 'invalid'
        
        
    def draw(self,
             vertical = True,
             barWidth = 0.9,
             colorsBarsFill = None,
             colorsBarsBorder = None,
             barsBorderWidth = 5,
             errBar = True,
             errBarCapWidth = 0.0,
             errBarColor = "black",
             errCapSize = 10,
             scatterData = True,
             scatterDataColor = [0,0,0,0.7],      
             scatterTickLabels = 'o', 
             groupBy = None, #should be a number
             groupByAxis = None,
             markers = None,
             annotate = True,
             annotateLabels = None):
        
        #todo: colors of ticklabels the same as default border labels, add annotations and ticklabels to the plot
        #### Make data ###
        y = self.raw_data
        x = list(range(len(y))) # x-coordinates of your bars
        
        
        
        ### Take into account all the parameters ###
        
     
          #if both borders and bar fills are not set, than defalut is no fill only borders
        #otherwise it will use to the colors set by user
        if colorsBarsFill is None and colorsBarsBorder is None:
            colorsBarsBorder = True
        
        if self._identify_colors(colorsBarsFill)=='invalid':
            colorsBarsFill = plt.cm.Set3(np.arange(len(self.raw_data)))
            
        if self._identify_colors(colorsBarsBorder)=='invalid':
            colorsBarsBorder = plt.cm.Set3(np.arange(len(self.raw_data)))
            
        
        if colorsBarsFill is None:
            colorsBarsFill = [0,0,0,0]
        if colorsBarsBorder is None:
            colorsBarsBorder = [0,0,0,0]
            
        if self._identify_colors(errBarColor)=='invalid':
            errBarColor = 'black'
      
        if hasattr(scatterTickLabels, "__len__") and not isinstance(scatterTickLabels, str) and len(scatterTickLabels)!=self.length:
            scatterTickLabels = 'o'
        
        
        #compute error Bars
        if errBar:
            stds = [np.std(yi) for yi in y]
        #compute bar heights
        bar_heights = [np.mean(yi) for yi in y]
        
        
        fig, ax = plt.subplots()
        
        ### Draw a vertical main plot ###
        if vertical:
            ax.bar(x,
                height=bar_heights,
                yerr=stds,    # error bars
                ecolor = errBarColor,
                capsize=errCapSize, # error bar cap width in points
                width=barWidth,    # bar width
                # tick_label=tickLabels,
                color=colorsBarsFill,  # face color transparent
                edgecolor=colorsBarsBorder,
                linewidth = barsBorderWidth,
            )
        ### Draw a horizonatal main plot ###
        else:
            ax.barh(x,
                bar_heights,
                height=barWidth,
                xerr=stds,    # error bars
                ecolor = errBarColor,
                capsize = errCapSize,
                # tick_label=tickLabels,
                color=colorsBarsFill,  # face color transparent
                edgecolor=colorsBarsBorder,
                linewidth = barsBorderWidth,
            )
        ## idea to to add scatterdata gradients, like a parameter gradient so the transparency of points depend on the value
        if scatterData:
            if vertical:
                for i in range(len(x)):
                    # # distribute scatter randomly across whole width of bar
                    #   y = self.raw_data
                    #   x = list(range(len(y))) # x-coordinates of your bars
        
                    ax.scatter(x[i] + np.random.random(y[i].size) * barWidth/4 - barWidth / 8, y[i], marker = scatterTickLabels,color=scatterDataColor)
                    # ax.scatter(np.full(y[i].size, x[i]), y[i], marker = 'v',color='black')

            else:
                for i in range(len(y)):
                    # distribute scatter randomly across whole width of bar
                    ax.scatter(y[i], x[i] + np.random.random(y[i].size) * barWidth/4 - barWidth / 8, marker = scatterTickLabels,color=scatterDataColor)



        plt.show()
    
    
    
    
    # def compose():
        
        
         # if groupBy: call the function itself
        # # markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
        # #also add subplots
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
# data = np.random.normal(100, 20, 200)
data = y = [np.random.random(30) * 20 + 5, np.random.random(30) * 20 + 8, np.random.random(30) * 20 + 2]
print(len(data[0]))
bar = BarGraph(data)
# Creating dataset
bar.draw(vertical = False, barWidth = 0.9, colorsBarsFill = 'red')


# use a decorator function to put what you want on the top of the barplots



# %%
# print(color.is_color_like('red'))
# print(color.is_color_like([0,0,0]))
# print(color.is_color_like([0,0]))


bar.draw(vertical = False, barWidth = 0.9,  barsBorderWidth = 10, scatterTickLabels = ['lol', 'hi'])


# %%
print(np.random.uniform(5))
# %%
