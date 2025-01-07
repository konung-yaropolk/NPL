import scsv as csv
import numpy as np
import matplotlib.pyplot as plt
import statlib



# Example usage:

# %%# generate random normal data:
groups = 2
data = [list(np.random.normal(.5*i + 3, 1-.2*i, 100)) for i in range(groups)]

# %%# generate random non-normal data:
# groups = 2
# data = [list(np.random.uniform(i+3, i+1, 100)) for i in range(groups)]


# %%# or load from csv:
# new_csv = csv.OpenFile('data.csv')
# data = new_csv.ColsFloat[2:4]

# %%# set the parameters:
paired = False   # is groups dependend or not
tails = 2        # two-tailed or one-tailed result
popmean = 0        # population mean - only for single-sample tests needed

# %%# initiate the analysis
analysis = statlib.StatisticalAnalysis(
    data, paired=paired, tails=tails, popmean=0)

# %%# Preform auto-selected test
analysis.RunAuto()


# %%# Preform scecific tests:

# # 2 groups independend:
# analysis.RunTtest()
# analysis.RunMannWhitney()

# # 2 groups paired
# analysis.RunTtestPaired()
# analysis.RunWilcoxon()

# # 3 and more indepennded groups comparison:
# analysis.RunAnova()
# analysis.RunKruskalWallis()

# # 3 and more paired groups comparison:
# analysis.RunFriedman()

# # single group test
# analysis.RunTtestSingleSample()
# analysis.RunWilcoxonSingleSample()


# %%# Get the results dictionary for future processing
results = analysis.GetResult()



# %%# Make Barplot
def barplot(data_samples, p=1, stars='ns', sd=0, mean=0, median=0):
    fig, ax = plt.subplots()

    colors = ['k', 'r', 'b', 'g', 'orange']

    for i, data in enumerate(data_samples):
        x = i + 1  # Bar position
        # Bars:
        ax.bar(x, 
               mean[i], 
               yerr=sd[i], 
               capsize=10, 
               edgecolor=colors[i % len(colors)], 
               fill=False, 
               linewidth=2)
        # Data points 
        spread = np.random.uniform(-0.1, 0.1, size=len(data))  # Adjust spread range
        ax.scatter(x + spread, data, color='black', s=16, zorder=3, alpha=0.5)
        ax.plot(x,
                median[i],
                marker='o', 
                markerfacecolor='none', 
                markeredgecolor='black', 
                markersize=10, 
                markeredgewidth=1)

    # Significance bar
    y_range = max([max(data) for data in data_samples])
    x1, x2 = 1, len(data_samples)
    y, h, col = 1.05 * y_range, .1 * y_range, 'k'
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5,
             y + h,
             '{}\n{}'.format(p, stars),
             ha='center',
             va='bottom',
             color=col)

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.xaxis.set_visible(False)

    plt.show()


barplot(data, 
        p=results['p-value'], 
        stars=results['Stars_Printed'], 
        sd=results['Groups_SD'], 
        mean=results['Groups_Mean'],
        median=results['Groups_Median'])



