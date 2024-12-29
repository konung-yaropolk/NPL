import CsvParser as csv
import numpy as np
import scipy.stats as stats


class __StatisticalTests():
    '''Statistical tests mixin'''

    def t_test(self):
        if self.paired:
            t_stat, t_p_value = stats.ttest_rel(
                self.data_list[0], self.data_list[1])
        else:
            t_stat, t_p_value = stats.ttest_ind(
                self.data_list[0], self.data_list[1])

        if self.tails == 1:
            t_p_value /= 2

        self.test_name = "t-test"
        self.test_stat = t_stat
        self.p_value = t_p_value

    def mann_whitney_u_test(self):
        stat, p_value = stats.mannwhitneyu(
            self.data_list[0], self.data_list[1], alternative='two-sided' if self.tails == 2 else 'greater')
        self.test_name = "Mann-Whitney U test"
        self.test_stat = stat
        self.p_value = p_value

    def wilcoxon_signed_rank_test(self):
        stat, p_value = stats.wilcoxon(self.data_list[0], self.data_list[1])
        if self.tails == 1 and p_value > 0.5:
            p_value = 1 - p_value
        self.test_name = "Wilcoxon signed-rank test"
        self.test_stat = stat
        self.p_value = p_value

    def anova(self):
        stat, p_value = stats.f_oneway(*self.data_list)
        if self.tails == 1 and p_value > 0.5:
            p_value /= 2
        self.test_name = "ANOVA"
        self.test_stat = stat
        self.p_value = p_value

    def kruskal_wallis_test(self):
        stat, p_value = stats.kruskal(*self.data_list)
        self.test_name = "Kruskal-Wallis test"
        self.test_stat = stat
        self.p_value = p_value

    def friedman_test(self):
        stat, p_value = stats.friedmanchisquare(*self.data_list)
        self.test_name = "Friedman test"
        self.test_stat = stat
        self.p_value = p_value


class __NormalityTests():
    '''Normality tests mixin'''

    def check_normality(self, data):
        # Shapiro-Wilk Test
        sw_stat, sw_p_value = stats.shapiro(data)
        if sw_p_value > 0.05:
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

        # Use critical values from the Lilliefors table
        # (two-sided test, approximations for large n)
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


class StatisticalAnalysis(__StatisticalTests, __NormalityTests):
    '''
        The main class
        *documentation placeholder*

    '''

    def __init__(self, data_list, paired=False, tails=2):
        self.data_list = data_list
        self.paired = paired
        self.tails = tails
        self.normals = []
        self.methods = []
        self.test_name = None
        self.test_stat = None
        self.p_value = None
        self.warning_flag_non_numeric_data = False

        # adjusting input data type
        self.data_list = self.__floatify_recursive(self.data_list)
        if self.warning_flag_non_numeric_data:
            print('\nWarnig: Non-numeric data was found in groups and ignored.')
            print('        Make sure the input data is correct to get the correct results\n')

        # Assertion block
        try:
            assert self.tails in [1, 2], "Tails parameter can be 1 or 2 only"
            assert len(
                self.data_list) > 1, "At least two groups of data must be given"
            assert all(len(
                lst) > 2 for lst in self.data_list), "Each group must contain at least three values"
            if self.paired == True:
                assert all(len(lst) == len(
                    self.data_list[0]) for lst in self.data_list), "Paired groups must be the same length"
        except AssertionError as error:
            print('\nError: ', error, '\n')
            exit()
            
        for data in self.data_list:
            normal, method = self.check_normality(data)
            self.normals.append(normal)
            self.methods.append(method)
            print(
                f"Data is {'normal' if normal else 'not normal'} (checked by {method})")

        self.__auto()
        self.result = self.__create_result_dict()

    def __auto(self):
        self.n_groups = len(self.data_list)
        self.parametric = all(self.normals)

        if self.n_groups == 2:
            if self.paired:
                if self.parametric:
                    return self.t_test()
                else:
                    return self.wilcoxon_signed_rank_test()
            else:
                if self.parametric:
                    return self.t_test()
                else:
                    return self.mann_whitney_u_test()
        else:
            if self.paired:
                return self.friedman_test()
            else:
                if self.parametric:
                    return self.anova()
                else:
                    return self.kruskal_wallis_test()

    def __floatify_recursive(self, data):
        if isinstance(data, list):
            # Recursively process sublists and filter out None values
            processed_list = [self.__floatify_recursive(item) for item in data]
            return [item for item in processed_list if item is not None]
        else:
            try:
                # Try to convert the item to float
                return np.float64(data)
            except (ValueError, TypeError):
                # If conversion fails, replace with None
                self.warning_flag_non_numeric_data = True
                return None
        
    def __make_stars(self):
        if self.p_value is not None:
            if self.p_value < 0.0001:
                return 4
            if self.p_value < 0.001:
                return 3
            elif self.p_value < 0.01:
                return 2
            elif self.p_value < 0.05:
                return 1
            else:
                return 0
        return 0

    def __make_p_value_printed(self):
        p = self.p_value.item()
        if p is not None:
            if p > 0.99:
                return "p > 0.99"
            elif p >= 0.01:
                return f"{p:.2g}"
            elif p >= 0.001:
                return f"{p:.3g}"
            elif p >= 0.0001:
                return f"{p:.4g}"
            else:
                return "p < 0.0001"
        return 'NaN'

    def __create_result_dict(self):

        self.stars_int = self.__make_stars()
        self.stars_str = '*' * self.stars_int if self.stars_int else 'ns'      

        return {
            "p-value" : self.__make_p_value_printed(),
            "StarsPrinted": self.stars_str,
            "TestName": self.test_name,
            "N_Groups": self.n_groups,
            "GroupSize": [len(self.data_list[i]) for i in range(len(self.data_list))],
            "ParametricStratistics": self.parametric,
            "PairedStratistics": self.paired,
            "Tails": self.tails,
            "p-value_exact": self.p_value.item(),
            "Stars":  self.stars_int,
            "Statistic": self.test_stat.item(),
        }

    def PrintResult(self):
        print('')
        for i in self.result:
            shift = 21 - len(i) 
            print(i, ':', ' ' *shift, self.result[i])    

    def GetResult(self):
        return self.result

    def GetStats(self):
        if self.test_stat is not None:
            return abs(self.test_stat)
        return None

    def GetP(self):
        return self.p_value

    def GetTestName(self):
        return self.test_name

    def GetStarsInt(self):
        return self.stars_int

    def GetStarsPrinted(self):
        return self.stars_str


# Example usage
#data = [list(np.random.normal(i, 1, 100)) for i in range(3)]

new_csv = csv.OpenFile('data.csv')
data = new_csv.Cols[0:4]


analysis = StatisticalAnalysis(data, paired=False, tails=2)
result = analysis.GetResult()
analysis.PrintResult()


