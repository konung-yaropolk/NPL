import SimpleCsv as csv
import numpy as np
import scipy.stats as stats


class __StatisticalTests():
    '''
        Statistical tests mixin
    '''

    def t_test(self):
        if self.paired:
            t_stat, t_p_value = stats.ttest_rel(
                self.groups_list[0], self.groups_list[1])
        else:
            t_stat, t_p_value = stats.ttest_ind(
                self.groups_list[0], self.groups_list[1])

        if self.tails == 1:
            t_p_value /= 2

        self.test_name = "t-test"
        self.test_stat = t_stat
        self.p_value = t_p_value

    def mann_whitney_u_test(self):
        stat, p_value = stats.mannwhitneyu(
            self.groups_list[0], self.groups_list[1], alternative='two-sided' if self.tails == 2 else 'greater')
        self.test_name = "Mann-Whitney U test"
        self.test_stat = stat
        self.p_value = p_value

    def wilcoxon_signed_rank_test(self):
        stat, p_value = stats.wilcoxon(self.groups_list[0], self.groups_list[1])
        if self.tails == 1 and p_value > 0.5:
            p_value = 1 - p_value
        self.test_name = "Wilcoxon signed-rank test"
        self.test_stat = stat
        self.p_value = p_value

    def anova(self):
        stat, p_value = stats.f_oneway(*self.groups_list)
        if self.tails == 1 and p_value > 0.5:
            p_value /= 2
        self.test_name = "ANOVA"
        self.test_stat = stat
        self.p_value = p_value

    def kruskal_wallis_test(self):
        stat, p_value = stats.kruskal(*self.groups_list)
        self.test_name = "Kruskal-Wallis test"
        self.test_stat = stat
        self.p_value = p_value

    def friedman_test(self):
        stat, p_value = stats.friedmanchisquare(*self.groups_list)
        self.test_name = "Friedman test"
        self.test_stat = stat
        self.p_value = p_value


class __NormalityTests():
    '''
        Normality tests mixin
    '''

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
    

class __TextFormatting():
    '''
        Text formatting mixin
    '''

    def print_groups(self, delimiter='                ', max_length=15):
        self.log('')
        # Get the number of groups (rows) and the maximum length of rows
        data = self.groups_list
        num_groups = len(data)
        max_len = max(len(row) for row in data)
        
        # Print the header
        header = [f"Group {i+1}" for i in range(num_groups)]
        space = [' '*7 for i in range(num_groups)]
        line = ['_'*7 for i in range(num_groups)]
        self.log(delimiter.join(header))
        self.log(delimiter.join(space))
        
        # Print each column with a placeholder if longer than max_length
        for i in range(max_len):
            row_values = []
            all_values_empty = True
            for row in data:
                if len(row) > max_length:
                    if i < max_length:
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    elif i == max_length:
                        row_values.append(f"[{len(row) - max_length} more]")
                        all_values_empty = False
                    else:
                        continue
                else:
                    if i < len(row):
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    else:
                        row_values.append('')
            if all_values_empty: break
            self.log(delimiter.join(row_values))

    def print_results(self):
        self.log('\n\nResults: \n')
        for i in self.results:
            shift = 21 - len(i) 
            self.log(i, ':', ' ' *shift, self.results[i])    
    
    def make_stars(self):
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

    def make_p_value_printed(self):
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

    def create_results_dict(self):

        self.stars_int = self.make_stars()
        self.stars_str = '*' * self.stars_int if self.stars_int else 'ns'      

        return {
            "p-value" : self.make_p_value_printed(),
            "StarsPrinted": self.stars_str,
            "TestName": self.test_name,
            "N_Groups": self.n_groups,
            "GroupSize": [len(self.groups_list[i]) for i in range(len(self.groups_list))],
            "ParametricStratistics": self.parametric,
            "PairedStratistics": self.paired,
            "Tails": self.tails,
            "p-value_exact": self.p_value.item(),
            "Stars":  self.stars_int,
            "Statistic": self.test_stat.item(),
        }


    def log(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        print(message, **kwargs)
        self.summary += '\n    ' + message


class __InputFormatting():
    def floatify_recursive(self, data):
        if isinstance(data, list):
            # Recursively process sublists and filter out None values
            processed_list = [self.floatify_recursive(item) for item in data]
            return [item for item in processed_list if item is not None]
        else:
            try:
                # Try to convert the item to float
                return np.float64(data)
            except (ValueError, TypeError):
                # If conversion fails, replace with None
                self.warning_flag_non_numeric_data = True
                return None


class StatisticalAnalysis(__StatisticalTests, __NormalityTests, __TextFormatting, __InputFormatting):
    '''
        The main class
        *documentation placeholder*

    '''

    def __init__(self, groups_list, paired=False, tails=2, test='auto'):
        self.groups_list = groups_list
        self.paired = paired
        self.tails = tails
        self.normals = []
        self.methods = []
        self.test_name = None
        self.test_stat = None
        self.p_value = None
        self.warning_flag_non_numeric_data = False
        self.summary = ''
        self.n_groups = len(self.groups_list)
        # self.__run_test(test)

    def __run_test(self, test='auto'):
        self.results = {}

        self.log("\n" + '-'*67)
        self.log("Statistics module initiated for data of {} groups\n".format(len(self.groups_list)))

        # adjusting input data type
        self.groups_list = self.floatify_recursive(self.groups_list)
        if self.warning_flag_non_numeric_data:
            self.log('\nWarnig: Non-numeric data was found in input and ignored.')
            self.log('        Make sure the input data is correct to get the correct results\n')

        # Assertion block
        try:
            assert self.tails in [1, 2], "Tails parameter can be 1 or 2 only"
            assert len(
                self.groups_list) > 1, "At least two groups of data must be given"
            assert all(len(
                lst) > 2 for lst in self.groups_list), "Each group must contain at least three values"
            assert not (self.paired == True and not all(len(lst) == len(
                    self.groups_list[0]) for lst in self.groups_list)), "Paired groups must be the same length"
            assert not (test == 'friedman' and not all(len(lst) == len(
                    self.groups_list[0]) for lst in self.groups_list)), "Paired groups must be the same length for Friedman Chi Square test"            
            assert not (test == 'wilcoxon' and not all(len(lst) == len(
                    self.groups_list[0]) for lst in self.groups_list)), "Paired groups must be the same length for Wilcoxon signed-rank test"     
            assert not (test == 'friedman' and len(self.groups_list) < 3), "At least three groups of data must be given for Friedman Chi Square test"
            assert not (test == 'wilcoxon' and len(self.groups_list) > 2), "Only two groups of data must be given for Wilcoxon signed-rank test"
        except AssertionError as error:
            print('\nError: ', error)
            self.log('-'*67 + '\n')
            return

        self.print_groups()

        self.log("\n\nNormality checked by both Shapiro-Wilk and Lilliefors tests")
        self.log("Group data is normal if at least one results is positive:\n")
        for i, data in enumerate(self.groups_list):
            normal, method = self.check_normality(data)
            self.normals.append(normal)
            self.methods.append(method)
            self.log(f"    Group {i+1}: disrtibution is {'normal' if normal else 'not normal'}")
        self.parametric = all(self.normals)

        self.log('\n')
        self.log('Parametric Statistics:', self.parametric)
        self.log('Paired Statistics:', self.paired)
        self.log('Groups:', self.n_groups)

        self.log('Test chosen by user:', test)

        if test == 'anova':
            self.anova()
        elif test == 'friedman':
            self.friedman_test() 
        elif test == 'kruskal_wallis':
            self.kruskal_wallis_test()   
        elif test == 'mann_whitney':
            self.mann_whitney_u_test()
        elif test == 't-test':
            self.t_test()
        elif test == 'wilcoxon':
            self.wilcoxon_signed_rank_test()
        else:
            self.log('Automatic test selection preformed.')
            self.__auto()  

        self.results = self.create_results_dict()
        self.print_results()
        self.log('\n\nResults above are accessible as a dictionary via GetResult() method')
        self.log('-'*67 + '\n')

    def __auto(self):

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

    def RunAuto(self):
        self.__run_test(test='auto') 

    def RunAnova(self):
        self.__run_test(test='anova')

    def RunFriedman(self):
        self.__run_test(test='friedman')

    def RunKruskalWallis(self):
        self.__run_test(test='kruskal_wallis')

    def RunMannWhitney(self):
        self.__run_test(test='mann_whitney')

    def RunTtest(self):
        self.__run_test(test='t-test')

    def RunWilcoxon(self):
        self.__run_test(test='wilcoxon')
    
    def GetResult(self):
        try:
            return self.results
        except AttributeError as error:
            print(error)
            return {}
    
    def GetSummary(self):
        return self.summary
    
    def PrintSummary(self):
        print(self.summary)




# Example usage

# data = [list(np.random.normal(i, 1, 100)) for i in range(3)]
# data = [list(np.random.uniform(i, 1, 100)) for i in range(3)]

new_csv = csv.OpenFile('data.csv')
data = new_csv.Cols[0:2]


analysis = StatisticalAnalysis(data, paired=False, tails=2)
analysis.RunAuto()
# analysis.RunFriedman()
# analysis.RunKruskalWallis()
# analysis.RunMannWhitney()
# analysis.RunTtest()
# analysis.RunWilcoxon()

results = analysis.GetResult()
