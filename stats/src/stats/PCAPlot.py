# %%
#!/usr/bin/env python
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Only use for plot layout adjustment
DEBUG = False
# %%


class PCAPlot:
    '''
        Input format:

        data: array of arrays, where each array is a feature. So the length of array of arrays is a number of features.

    '''
    # add 3D plots for pcs, plotting eigenvectors, explained variance with bars instead of a line plot

    def __init__(self,
                 data: list,
                 feature_names: list,
                 mask=None) -> None:

        self.raw_data = data
        self.mask = mask
        self.feature_names = np.array(feature_names)
        self.normalized_data = self.normalize_data()
        self.n_features = len(feature_names)
        self.PCA = self.get_fitted_PCA()
        self.principal_components = self.PCA.components_
        self.transformed_data = self.transform_data()
        self.explained_variance = self.PCA.explained_variance_ratio_
        self.eigenvalues = self.PCA.explained_variance_
        self.covariance = self.PCA.get_covariance()
        # assert that featrues are the same length as data

    # helper methods

    def normalize_data(self) -> list:
        sc = StandardScaler()
        return sc.fit_transform(self.raw_data)

    def get_fitted_PCA(self):
        pca = PCA().fit(self.normalized_data)
        return pca

    def transform_data(self):
        return self.PCA.transform(self.normalized_data)

   # plotting methods

    def plot_original_data_feature_matrix(self):
        df = pd.DataFrame(self.raw_data, columns=self.feature_names)

        fig = px.scatter_matrix(df, color=self.mask)
        fig.update_traces(diagonal_visible=False)
        fig.show()

    def plot_covariance_matrix(self,
                               annot=False,
                               cmap='coolwarm',
                               title='Covariance Matrix Heatmap'):
        sns.heatmap(self.covariance, annot=annot, cmap=cmap,
                    xticklabels=self.feature_names, yticklabels=self.feature_names)
        plt.title(title)
        plt.show()

    def plot_principal_components_heatmap(self,
                                          principal_components='all',
                                          features='all',
                                          annot=True,
                                          cmap='coolwarm',
                                          title='Principal Components Heatmap'):
        if principal_components == 'all':
            principal_components = range(self.n_features)
        else:
            principal_components = list(
                map(lambda x: x-1, principal_components))
        if features == 'all':
            features = range(self.n_features)
        else:
            features = list(map(lambda x: x-1, features))

        sns.heatmap(self.principal_components[principal_components].T[features], annot=True, cmap=cmap, xticklabels=list(
            map(lambda x: f'PC{x+1}', principal_components)), yticklabels=self.feature_names[features])
        plt.title(title)
        plt.show()

    def plot_explained_variance(self):
        plt.plot(range(1, len(self.explained_variance) + 1),
                 self.explained_variance, marker='o', linestyle='--')
        plt.title('')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.show()

    def plot_cumulative_explained_variance(self,
                                           threshold=0.9):
        # Calculate cumulative explained variance
        cumulative_explained_variance = np.cumsum(self.explained_variance)

        plt.plot(range(1, len(cumulative_explained_variance) + 1),
                 cumulative_explained_variance, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=threshold, color='r', linestyle='-')
        plt.text(0.5, 0.85, f'{threshold*100}% cut-off threshold', color='red')
        plt.show()

    def plot_transformed_data_2D(self,
                                 x_axis=1,
                                 y_axis=2,
                                 loadings=[],
                                 label_points=False):
        fig, ax = plt.subplots()
        if len(loadings):
            load_vecs = self.principal_components.T * np.sqrt(self.eigenvalues)
            for i in loadings:
                plt.arrow(0, 0, load_vecs[i-1, x_axis-1], load_vecs[i-1, y_axis-1], head_width=0.03,
                          head_length=0.06, linewidth=0.1, length_includes_head=True, color='r')
                plt.text(load_vecs[i-1, x_axis-1] * 1.15, load_vecs[i-1, y_axis-1]
                         * 1.15, "Var"+str(i), color='g', ha='center', va='center')

        # labeling x and y axes
        PCX, PCY = self.transformed_data[:, x_axis -
                                         1], self.transformed_data[:, y_axis-1]
        scalePCX = 1.0/(PCX.max() - PCX.min())
        scalePCY = 1.0/(PCY.max() - PCY.min())

        if label_points:
            ax.scatter(PCX*scalePCX, PCY*scalePCY, s=5)
            for i in range(len(PCX)):
                ax.text(PCX[i] * scalePCX,
                        PCY[i] * scalePCY, str(i+1),
                        fontsize=10)
        else:
            ax.scatter(PCX*scalePCX, PCY*scalePCY)

        ax.set_xlabel(f'Principal Component {x_axis}')
        ax.set_ylabel(f'Principal Component {y_axis}')
        plt.show()

    def plot_transformed_data_3D(self,
                                 x_axis=1,
                                 y_axis=2,
                                 z_axis=3,
                                 loadings=[],
                                 label_points=False):
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot(111, projection='3d')
        PCX, PCY, PCZ = self.transformed_data[:, x_axis -
                                              1], self.transformed_data[:, y_axis-1], self.transformed_data[:, z_axis-1]
        scalePCX = 1.0/(PCX.max() - PCX.min())
        scalePCY = 1.0/(PCY.max() - PCY.min())
        scalePCZ = 1.0/(PCZ.max() - PCZ.min())
        if label_points:
            axis.scatter(PCX*scalePCX, PCY*scalePCY, PCZ*scalePCZ, s=5)
            for i in range(len(PCX)):
                axis.text(PCX[i] * scalePCX,
                          PCY[i] * scalePCY,
                          PCZ[i] * scalePCZ,
                          str(i+1),
                          fontsize=10)
        else:
            axis.scatter(PCX*scalePCX, PCY*scalePCY, PCZ*scalePCZ)

        if len(loadings):
            load_vecs = self.principal_components.T * np.sqrt(self.eigenvalues)
            # scalePC1 = 1.0/(PC1.max() - PC1.min())
            # scalePC2 = 1.0/(PC2.max() - PC2.min())
            for i in loadings:
                axis.quiver(
                    0, 0, 0,  # <-- starting point of vector
                    # <-- directions of vector
                    load_vecs[i-1, x_axis-1], load_vecs[i-1,
                                                        y_axis-1], load_vecs[i-1, z_axis-1],
                    color='red', alpha=.8, lw=2, arrow_length_ratio=0.1
                )
                axis.text(load_vecs[i-1, x_axis-1], load_vecs[i-1, y_axis-1],
                          load_vecs[i-1, z_axis-1], "Var"+str(i), color='green')

                # axis.plot([0, load_vecs[i-1, x_axis-1]], [0,load_vecs[i-1, y_axis-1]], [0,load_vecs[i-1, z_axis-1]], color = 'r')
                # plt.text(load_vecs[i-1,x_axis-1]* 1.15, load_vecs[i-1,y_axis-1] * 1.15, load_vecs[i-1,z_axis-1] * 1.15, "Var"+str(i), color = 'g', ha = 'center', va = 'center')

        # labeling x and y axes
        axis.set_xlabel(f'Principal Component {x_axis}')
        axis.set_ylabel(f'Principal Component {y_axis}')
        axis.set_zlabel(f'Principal Component {z_axis}')

    def plot_transformed_data_matrix(self,
                                     num_of_components='all'):
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(self.explained_variance * 100)
        }
        if num_of_components == 'all':
            num_of_components = self.n_features
        fig = px.scatter_matrix(
            self.transformed_data,
            labels=labels,
            dimensions=range(num_of_components))
        fig.update_traces(diagonal_visible=False)
        fig.show()

  # analytical methods
    def print_features_of_each_component(self):
        data = pd.DataFrame(pca.principal_components.T, columns=[
                            f'PC{i}' for i in range(1, self.n_features+1)], index=self.feature_names)
        print(data)

    def print_important_features_for_each_component(self,
                                                    threshold=0.3):
        for i in range(self.n_features):
            filter = abs(self.principal_components[i]) > threshold
            print(f"PC{i+1}: {', '.join(self.feature_names[filter])}")

    def print_important_components_kaiser(self):
        # Assuming pca is already fitted PCA object from sklearn
        for i in range(len(self.eigenvalues)):
            if self.eigenvalues[i] > 1:
                print(i+1)


# %%


data = load_wine().data
features = load_wine().feature_names
# %%
data = load_diabetes().data
features = load_diabetes().feature_names
pca = PCAPlot(data, features)
# %%
pca = PCAPlot(data, features)

pca.plot_transformed_data_2D(1, 2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], True)

# %%
# (num of samples, num of features)
# cov matrix is (num of features * num of features)
# threre are 13 principal components of length 13 and 13 eigenvectors
pca = PCAPlot(data, features)
# pca.print_features_of_each_component()

# pca.print_important_features_for_each_component()
# pca.plot_principal_components_heatmap()
# pca.plot_cumulative_explained_variance()

# pca.print_important_components_kaiser()

# pca.plot_any_two_normalized_features_with_eigenvectors()
# pca.plot_principal_components_heatmap([1,2])

pca.plot_transformed_data_3D(1, 3, 5, [1, 2, 3, 4, 5, 6], True)
# pca.plot_original_data_feature_matrix()
# pca.plot_transformed_data_matrix()
# pca.plot_transformed_data_2D(1, 2, [1,2,3,4,5,6,7,8,9,10,11,12, 13])
# %%
# plt.plot(norm, '-gs')
pca.plot_covariance_matrix()
# %%
pca.plot_any_two_normalized_features_with_eigenvectors()

# %%
format(pca_breast.explained_variance_ratio_)

# %%
all = [1, 1]
print([2, 4, 5, 7, 5, 3, 2, 6, 7, 4, 3, 6, 4, 3, 63][all])

# %%
# sources:
# sklearn docs https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
#     1. https://drlee.io/secrets-of-pca-a-comprehensive-guide-to-principal-component-analysis-with-python-and-colab-6f7f3142e721
# 2. https://www.geeksforgeeks.org/principal-component-analysis-pca/
# 3. https://colab.research.google.com/drive/1w4OJqS3pgtrzViBWGM9i7r3QYNbanMwq?usp=sharing
# 4. https://www.datacamp.com/tutorial/principal-component-analysis-in-python
# 5. https://www.geeksforgeeks.org/kmeans-clustering-and-pca-on-wine-dataset/
# 6. https://www.geeksforgeeks.org/principal-component-analysis-with-python/
# 7. https://pub.towardsai.net/covariance-matrix-visualization-using-seaborns-heatmap-plot-64332b6c90c5
# 8. https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/

# https://stackoverflow.com/questions/57340166/how-to-plot-the-pricipal-vectors-of-each-variable-after-performing-pca
# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-a-3d-plot
# https://plotly.com/python/pca-visualization/

# https://stackoverflow.com/questions/57340166/how-to-plot-the-pricipal-vectors-of-each-variable-after-performing-pca
# https://www.reneshbedre.com/blog/principal-component-analysis.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

# PCA library for plotting
# https://erdogant.github.io/pca/pages/html/Examples.html

# using scikit for other stuff https://www.geeksforgeeks.org/learning-model-building-scikit-learn-python-machine-learning-library/
# https://statisticsglobe.com/biplot-pca-python
# https://www.jcchouinard.com/pca-plot-visualization-python/
# https://machinelearningmastery.com/principal-component-analysis-for-visualization/
# https://towardsdatascience.com/principal-component-analysis-fbce2a22c6e0
# https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
# https://towardsdatascience.com/pca-a-practical-journey-preprocessing-encoding-and-inspiring-applications-64371cb134a
# https://towardsdatascience.com/principal-components-analysis-plot-for-python-311013a33cd9
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py
