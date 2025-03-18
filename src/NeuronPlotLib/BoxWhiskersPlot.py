# %%
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

# Only use for plot layout adjustment
DEBUG = False
# %%

# !!!! in newer versions of matplotlib vert turns into "orientantation" and
# needs a rewrite
# ідея імпортувати ці точечки щоб замість них могли робити щось інше прикольне типу маленьких мишей


class BoxWhiskersPlot:
    def __init__(self,
                 data: list,
                 mask=None) -> None:

        self.raw_data = data
        self.length = len(data)
    # Whether to draw a notched boxplot (True), or a rectangular boxplot (False). The notches represent the confidence interval (CI) around the median. The documentation for bootstrap describes how the locations of the notches are computed by default, but their locations may also be overridden by setting the conf_intervals parameter.

    def draw(self,
             positions=None,  # positions of boxes, defaults to range(1,n+1)
             widths=None,
             tickLabels=None,
             notch=False,
             confidences=None,
             fliers=False,
             fliersMarker='',
             flierFillColor=None,
             flierEdgeColor=None,
             flierLineWidth=None,
             flierLineStyle=None,
             vertical=True,
             # whiskers when one float is tukeys parameter, when a pair of percentages, defines the percentiles where the whiskers should be If a float, the lower whisker is at the lowest datum above Q1 - whis*(Q3-Q1), and the upper whisker at the highest datum below Q3 + whis*(Q3-Q1), where Q1 and Q3 are the first and third quartiles. The default value of whis = 1.5 corresponds to Tukey's original definition of boxplots.
             whiskers=1.5,
             bootstrap=None,
             whiskersColor=None,
             whiskersLineWidth=None,
             whiskersLineStyle=None,
             showWhiskersCaps=True,
             whiskersCapsWidths=None,
             whiskersCapsColor=None,
             whiskersCapsLineWidth=None,
             whiskersCapsLineStyle=None,
             boxFill=None,
             boxBorderColor=None,
             boxBorderWidth=None,
             userMedians=None,
             medianColor=None,
             medianLineStyle=None,
             medianLineWidth=None,
             showMeans=False,
             meanMarker=None,
             meanFillColor=None,
             meanEdgeColor=None,
             meanLine=False,
             meanLineColor=None,
             meanLineStyle=None,
             meanLineWidth=None,
             autorange=False):
        if (not hasattr(positions, "__len__") or
            len(positions) != self.length or
                any(not isinstance(x, (int, float)) for x in positions)):
            positions = None
        if fliers == False:
            fliersMarker = ""
        else:
            if fliersMarker == "":
                fliersMarker = 'b+'
        # write a function to make a dictionary
        whiskersCapsStyles = dict()
        if whiskersCapsColor != None:
            whiskersCapsStyles["color"] = whiskersCapsColor
        if whiskersCapsLineWidth != None:
            whiskersCapsStyles["linewidth"] = whiskersCapsLineWidth
        if whiskersCapsLineStyle != None:
            whiskersCapsStyles['linestyle'] = whiskersCapsLineStyle

        boxProps = {"facecolor": (0, 0, 0, 0),
                    "edgecolor": "black", "linewidth": 1}
        if boxFill != None:
            boxProps["facecolor"] = boxFill
        if boxBorderColor != None:
            boxProps["edgecolor"] = boxBorderColor
        if boxBorderWidth != None:
            boxProps['linewidth'] = boxBorderWidth
        # if boxBorderStyle != None:
        #     boxProps['linestyle'] = boxBorderStyle  !!!this feature is not working with patch_artist that is needed for facecolor to work

        whiskersProps = {"color": 'black',
                         "linestyle": "solid", "linewidth": 1}
        if whiskersColor != None:
            whiskersProps["color"] = whiskersColor
        if whiskersLineStyle != None:
            whiskersProps["linestyle"] = whiskersLineStyle
        if whiskersLineWidth != None:
            whiskersProps['linewidth'] = whiskersLineWidth

        flierProps = {"markerfacecolor": [
            0, 0, 0, 0], "markeredgecolor": "black", "linestyle": "solid", "markeredgewidth": 1}
        if flierFillColor != None:
            flierProps["markerfacecolor"] = flierFillColor
        if flierEdgeColor != None:
            flierProps["markeredgecolor"] = flierEdgeColor
        if flierLineWidth != None:
            flierProps['markeredgewidth'] = flierLineWidth
        if flierLineStyle != None:
            flierProps['linestyle'] = flierLineStyle
        medianProps = {"linestyle": 'solid', "linewidth": 1, "color": 'red'}
        if medianColor != None:
            medianProps["color"] = medianColor
        if medianLineStyle != None:
            medianProps["linestyle"] = medianLineStyle
        if medianLineWidth != None:
            medianProps['linewidth'] = medianLineWidth

        meanProps = {"color": "black", "marker": 'o', "markerfacecolor": "black",
                     "markeredgecolor": "black", "linestyle": "solid", "linewidth": 1}

        if meanMarker != None:
            meanProps['marker'] = meanMarker
        if meanFillColor != None:
            meanProps["markerfacecolor"] = meanFillColor
        if meanEdgeColor != None:
            meanProps['markeredgecolor'] = meanEdgeColor
        if meanLineColor != None:
            meanProps["color"] = meanLineColor
        if meanLineStyle != None:
            meanProps['linestyle'] = meanLineStyle
        if meanLineWidth != None:
            meanProps['linewidth'] = meanLineWidth
        fig, ax = plt.subplots()
        ### Draw a vertical main plot ###
        ax.boxplot(self.raw_data,
                   positions=positions,
                   widths=widths,
                   # tick_labels=tickLabels,
                   notch=notch,
                   conf_intervals=confidences,
                   sym=fliersMarker,
                   flierprops=flierProps,
                   vert=vertical,
                   whis=whiskers,
                   whiskerprops=whiskersProps,
                   showcaps=showWhiskersCaps,
                   capwidths=whiskersCapsWidths,
                   capprops=whiskersCapsStyles,
                   boxprops=boxProps,
                   usermedians=userMedians,
                   medianprops=medianProps,
                   bootstrap=bootstrap,
                   showmeans=showMeans,
                   meanline=meanLine,
                   meanprops=meanProps,
                   autorange=autorange,
                   patch_artist=True)


# %%
data = y = [np.random.random(
    30) * 30 + 5, np.random.random(30) * 20 + 8, np.random.random(30) * 20 + 2]
box = BoxWhiskersPlot(data)
box.draw(fliers=True, autorange=True, meanLine=True)

# %%
fig, ax = plt.subplots()


# %%
boxprops = {"facecolor": "C0", "edgecolor": "white",
            "linewidth": 0.5

            boxprops = dict(
                linestyle='--', linewidth=3, color='darkgoldenrod')
            flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                              markeredgecolor='none')
            medianProps = dict(
                linestyle='-.', linewidth=2.5, color='firebrick')
            ax.boxplot(data_A, positions=pos - 0.2, patch_artist=True, label='Box A',
                       boxprops={'facecolor': 'steelblue'})
            ax.boxplot(data_B, positions=pos + 0.2, patch_artist=True, label='Box B',
                       boxprops={'facecolor': 'lightblue'})
            # Boxplots
            boxplot.boxprops.color: white
            boxplot.capprops.color: white
            boxplot.flierprops.color: white
            boxplot.flierprops.markeredgecolor: white
            mpl.rcParams['boxplot.flierprops.markeredgecolor'] = 'k'
            mpl.rcParams['boxplot.boxprops.color'] = 'b'
            mpl.rcParams['boxplot.whiskerprops.color'] = 'b'
            boxplot.flierprops.markeredgecolor: 'k'
            boxplot.boxprops.color:             'b'
            boxplot.whiskerprops.color:         'b'

            whiskerprops = {"color": "C0", "linewidth": 1.5},
            "boxplot.whiskerprops.color":     validate_color,
            "boxplot.whiskerprops.linewidth": validate_float,
            "boxplot.whiskerprops.linestyle": _validate_linestyle,

            flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                              markeredgecolor='none')
            medianprops = dict(
                linestyle='-.', linewidth=2.5, color='firebrick')
            meanpointprops = dict(marker='D', markeredgecolor='black',
                                  boxplot.flierprops.color: white
                                  boxplot.flierprops.markeredgecolor: white
                                  "boxplot.flierprops.color":           validate_color,
                                  "boxplot.flierprops.marker":          _validate_marker,
                                  "boxplot.flierprops.markerfacecolor": validate_color_or_auto,
                                  "boxplot.flierprops.markeredgecolor": validate_color,
                                  mpl.rcParams['boxplot.flierprops.color']='k'
                                  mpl.rcParams['boxplot.flierprops.marker']='+'
                                  mpl.rcParams['boxplot.flierprops.markerfacecolor']='none'
                                  mpl.rcParams['boxplot.flierprops.markeredgecolor']='k'
                                  boxplot.flierprops.color: b
                                  boxplot.flierprops.linestyle: none
                                  boxplot.flierprops.linewidth: 1.0
                                  boxplot.flierprops.marker: +
                                  flierprops=dict(linestyle='none', marker='d', mfc='g')))
'boxplot.flierprops.color': 'b',
'boxplot.flierprops.marker': 'o',


# sphx-glr-gallery-statistics-boxplot-py
https: // matplotlib.org/stable/gallery/statistics/boxplot.html
