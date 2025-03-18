# %%
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
data = y = np.random.normal(170, 10, 250)

# Only use for plot layout adjustment
DEBUG = False
# %%
class HistCumPlot:
    def __init__(self,
                 data: list) -> None:

        self.raw_data = data
        self.length = len(data)
    # Whether to draw a notched boxplot (True), or a rectangular boxplot (False). The notches represent the confidence interval (CI) around the median. The documentation for bootstrap describes how the locations of the notches are computed by default, but their locations may also be overridden by setting the conf_intervals parameter.

    def draw(self,
             bins = 'auto',
             color = None,
             cumulative = False,
             range = None,
             probabilityDensity = False,
             weights = None,
             bottomsBinLocations = None,
             type= 'bar', #bar, barstacked, step, stepfilled
             alignment = "mid", #can be also left, right,
             vertical = True,
             barWidth=1,
             logScale = False
             ):
        if range==None:
            range = [min(self.raw_data), max(self.raw_data)]
        if vertical==True:
            orientation = 'vertical'
        else:
            orientation = 'horizontal'
        fig, ax = plt.subplots()
        ### Draw a vertical main plot ###
        ax.hist(self.raw_data,
                color = color,
                bins = bins,
                range = range,
                density=probabilityDensity,
                weights = weights,
                cumulative = cumulative,
                bottom=bottomsBinLocations,
                histtype=type,
                align = alignment,
                orientation=orientation,
                rwidth=barWidth,
                log = logScale)


hist = HistCumPlot(data)
hist.draw(color = 'red')

