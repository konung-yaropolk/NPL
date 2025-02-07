import AutoStatLib
import matplotlib.pyplot as plt
import numpy as np

class StatsBarPlot():


    def barplot(self, data_samples, p=1, stars='ns', sd=0, mean=0, median=0, testname='', n=0):
        fig, ax = plt.subplots(figsize=(3, 4))

        colors = ['k', 'r', 'b', 'g']
        colors_fill = ['#CCCCCC', '#FFCCCC', '#CCCCFF', '#CCFFCC']

        for i, data in enumerate(data_samples):
            x = i + 1  # Bar position
            # Bars:
            ax.bar(x,
                mean[i],
                yerr=sd[i],
                width=.8,
                capsize=10,
                ecolor='r',
                edgecolor=colors[i % len(colors)],
                facecolor=colors_fill[i % len(colors_fill)],
                fill=True,
                linewidth=2)
            # Data points
            # Adjust spread range
            spread = np.random.uniform(-.10, .10, size=len(data))
            ax.scatter(x + spread, data, color='black', s=16, zorder=1, alpha=0.5)
            ax.plot(x,
                    median[i],
                    marker='x',
                    markerfacecolor='#00000000',
                    markeredgecolor='r',
                    markersize=10,
                    markeredgewidth=1)

        # Significance bar
        y_range = max([max(data) for data in data_samples])
        x1, x2 = 1, len(data_samples)
        y, h, col = 1.05 * y_range, .05 * y_range, 'k'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        ax.text((x1 + x2) * .5,
                y + h,
                '{}\n{}'.format(p, stars),
                ha='center',
                va='bottom',
                color=col)

        # Add subtitle aligned to the right at the bottom
        fig.text(0.95, 0.01, '{}\nn={}'.format(testname, str(n)[1:-1]),
                ha='right', va='bottom', fontsize=8)

        # Remove borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.xaxis.set_visible(False)

        plt.show()




groups = 2
n = 30
data = [list(np.random.normal(.5*i + 4, abs(1-.2*i), n))
        for i in range(groups)]

# set the parameters:
paired = False   # is groups dependend or not
tails = 2        # two-tailed or one-tailed result
popmean = 0        # population mean - only for single-sample tests needed

# initiate the analysis
analysis = AutoStatLib.StatisticalAnalysis(
    data, paired=paired, tails=tails, popmean=0)

analysis.RunAuto()
results = analysis.GetResult()

plot = StatsBarPlot

plot.barplot(data,data,
        p=results['p-value'],
        stars=results['Stars_Printed'],
        sd=results['Groups_SD'],
        mean=results['Groups_Mean'],
        median=results['Groups_Median'],
        testname=results['Test_Name'],
        n=results['Groups_N'],
        )