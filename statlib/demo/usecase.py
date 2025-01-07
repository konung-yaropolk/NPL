import scsv as csv
import numpy as np
import statlib


# Example usage:

# %%# generate random normal data:
groups = 2
data = [list(np.random.normal(i, 1, 100)) for i in range(groups)]

# %%# generate random non-normal data:
# groups = 2
# data = [list(np.random.uniform(i, i+10, 100)) for i in range(groups)]


# %%# or load from csv:
# new_csv = csv.OpenFile('data.csv')
# data = new_csv.Cols[2:4]

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

# results = analysis.GetResult()
