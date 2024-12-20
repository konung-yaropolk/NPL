import pandas as pd
import numpy as np
import scipy.stats as stats


class StatisticalAnalysis:
    def __init__(self, data_list, paired=False, tail='two-tailed'):
        self.data_list = data_list
        self.paired = paired
        self.tail = tail
        self.normals = []
        self.methods = []
        self.test_name = None
        self.test_stat = None
        self.test_p_value = None

        for data in data_list:
            normal, method = self.check_normality(data)
            self.normals.append(normal)
            self.methods.append(method)
            print(f"Data is {'normal' if normal else 'not normal'} (checked by {method})")

    def check_normality(self, data):
        # Shapiro-Wilk Test
        stat, p_value = stats.shapiro(data)
        if p_value > 0.05:
            return True, "Shapiro-Wilk"
        
        # Lilliefors Test (Kolmogorov-Smirnov test)
        lilliefors_stat, lilliefors_p_value = self.lilliefors_test(data)
        if lilliefors_p_value > 0.05:
            return True, "Lilliefors"

        return False, "Not normal"

    def lilliefors_test(self, data):
        ''' Lilliefors Test (Kolmogorov-Smirnov test) '''
        data = np.sort(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        cdf_values = stats.norm.cdf(data, loc=mean, scale=std)
        empirical_cdf = np.arange(1, n + 1) / n
        D_plus = np.max(empirical_cdf - cdf_values)
        D_minus = np.max(cdf_values - (np.arange(n) / n))
        D = max(D_plus, D_minus)

        # Use critical values from the Lilliefors table (two-sided test, approximations for large n)
        critical_values = {
            100: 0.072,
            1000: 0.024,
            5000: 0.018,
            10000: 0.015,
            np.inf: 0.01
        }

        for size, critical_value in critical_values.items():
            if n <= size:
                return D, 1 - critical_value if D < critical_value else 0

        return D, 0

    def t_test(self):
        if self.paired:
            t_stat, t_p_value = stats.ttest_rel(self.data_list[0], self.data_list[1])
        else:
            t_stat, t_p_value = stats.ttest_ind(self.data_list[0], self.data_list[1])
        
        if self.tail == 'one-tailed':
            t_p_value /= 2
        
        self.test_name = "t-test"
        self.test_stat = t_stat
        self.test_p_value = t_p_value

        return self._create_result_dict()

    def mann_whitney_u_test(self):
        u_stat, u_p_value = stats.mannwhitneyu(self.data_list[0], self.data_list[1], alternative='two-sided' if self.tail == 'two-tailed' else 'greater')
        self.test_name = "Mann-Whitney U test"
        self.test_stat = u_stat
        self.test_p_value = u_p_value

        return self._create_result_dict()

    def wilcoxon_signed_rank_test(self):
        w_stat, w_p_value = stats.wilcoxon(self.data_list[0], self.data_list[1])
        if self.tail == 'one-tailed' and w_p_value > 0.5:
            w_p_value = 1 - w_p_value
        self.test_name = "Wilcoxon signed-rank test"
        self.test_stat = w_stat
        self.test_p_value = w_p_value

        return self._create_result_dict()

    def anova(self):
        f_stat, p_value = stats.f_oneway(*self.data_list)
        if self.tail == 'one-tailed' and p_value > 0.5:
            p_value /= 2
        self.test_name = "ANOVA"
        self.test_stat = f_stat
        self.test_p_value = p_value

        return self._create_result_dict()

    def kruskal_wallis_test(self):
        f_stat, p_value = stats.kruskal(*self.data_list)
        self.test_name = "Kruskal-Wallis test"
        self.test_stat = f_stat
        self.test_p_value = p_value

        return self._create_result_dict()

    def friedman_test(self):
        f_stat, p_value = stats.friedmanchisquare(*self.data_list)
        self.test_name = "Friedman test"
        self.test_stat = f_stat
        self.test_p_value = p_value

        return self._create_result_dict()

    def auto(self):
        n_groups = len(self.data_list)
        all_normal = all(self.normals)

        if n_groups == 2:
            if self.paired:
                if all_normal:
                    return self.t_test()
                else:
                    return self.wilcoxon_signed_rank_test()
            else:
                if all_normal:
                    return self.t_test()
                else:
                    return self.mann_whitney_u_test()
        else:
            if self.paired:
                return self.friedman_test()
            else:
                if all_normal:
                    return self.anova()
                else:
                    return self.kruskal_wallis_test()


    def GetSigmas(self):
        if self.test_stat is not None:
            return abs(self.test_stat)
        return None

    def GetP(self):
        return self.test_p_value

    def GetTestName(self):
        return self.test_name

    def _create_result_dict(self):
        return {
            "TestName": self.test_name,
            "Statistic": self.test_stat.item(),
            "p-value": self.test_p_value.item(),
        }

# Example usage
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1, 100)
data3 = np.random.normal(2, 1, 100)

analysis = StatisticalAnalysis([data1, data2], paired=False, tail='two-tailed')

# Running the auto method to automatically decide the test
results = analysis.auto()
print(f"{results['TestName']} result: statistic={results['Statistic']}, p-value={results['p-value']}")

# Accessing separate attributes:
sigmas = analysis.GetSigmas()
p_value = analysis.GetP()
test_name = analysis.GetTestName()

print(f"Number of sigmas: {sigmas}")
print(f"P-value: {p_value}")
print(f"Test name: {test_name}")
