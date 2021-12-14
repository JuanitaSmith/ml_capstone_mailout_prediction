# suppress matplotlib user warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# for creating a feature tree
from xgboost import plot_tree

from sklearn.preprocessing import MinMaxScaler

from src.config import *

#
# Display inline matplotlib plots with IPython
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='white', font_scale=1)


def plot_features(column, df, title='Unique values and distribution',
                  figsize=(18, 30), rect=[0, 0.03, 1, 0.95], fontsize=14, logscale=False):
    """
    This function plots a count plot showing the distribution of unique values of each columns
    """

    sns.set(style='white', font_scale=1)
    fig = plt.figure(figsize=figsize)

    if logscale:
        plt.yscale('log')

    for i, column in enumerate(column):
        ax = fig.add_subplot(12, 3, i + 1)
        sns.countplot(x=df[column], ax=ax);

    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout(rect=rect)


def display_correlations(corr_matrix, figsize=(30, 30), title='Correlation Heatmap', dpi=600):
    """ Shows the top correlated features in a more condense triangle"""
    sns.set(style='white', font_scale=3)

    # display shows all of a dataframe
    plt.subplots(figsize=figsize, dpi=dpi)

    # hide the upper triangle, as it's the same as bottom triagle there is no need to show it
    # create array of zero's the same shape and type as the data
    mask = np.zeros_like(corr_matrix, dtype=bool)
    # triu_indices_from returns the indices for the upper triangle of the data
    mask[np.triu_indices_from(mask)] = True

    cmap = "RdBu"
    s = sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,
                    cmap=cmap,
                    cbar=False,
                    linecolor="grey",
                    square=True,
                    #                     center=0,
                    linewidths=1,
                    #                     cbar_kws={'shrink':0.3,
                    #                               'ticks': [-1, -.5, 0., 0.5, 1]},
                    vmin=-1,
                    vmax=1,
                    annot_kws={"size": 12})

    plt.title(title, fontsize=20)
    s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
    s.set_xticklabels(s.get_xticklabels(), rotation=90, fontsize=12)

    plt.show()

    sns.set(style='white', font_scale=1)


def scree_plot(pca, annotate=True, cumulative=True):
    """  Investigate the variance accounted for by each principal component """
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(15, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    if cumulative:
        ax.plot(ind, cumvals)

    if annotate:
        for i in range(num_components):
            ax.annotate(r"%s%%" % ((str(vals[i] * 100)[:4])), (ind[i] + 0.2, vals[i]), va="bottom", ha="center",
                        fontsize=12)

        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


def compare_distributions(df1, df2, columns):
    """ compare distribution of a column in 2 different datasets"""
    for i, column in enumerate(columns):
        fig, axis = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

        axis[0].set_title('Few Missing Values')
        axis[1].set_title('Lots Missing Values')
        sns.histplot(df1[column][df1[column].notnull()], label='few missing',
                     kde=True, color='b', ax=axis[0])
        sns.histplot(df2[column][df2[column].notnull()], label='lots missing',
                     kde=True, color='r', ax=axis[1])


def compare_distributions_3by3(df1, df2, df3, titles, columns, kde=False, sharey=True, sharex=True,
                               stat='count', logscale=False):
    """
    Compare distribution of a column in 3 different datasets

    count: show the number of observations in each bin
    frequency: show the number of observations divided by the bin width
    probability: or proportion: normalize such that bar heights sum to 1
    percent: normalize such that bar heights sum to 100
    density: normalize such that the total area of the histogram equals 1
    """

    sns.set(style='white', font_scale=1)

    for i, column in enumerate(columns):
        fig, axis = plt.subplots(1, 3, figsize=(16, 4), sharey=sharey, sharex=sharex, squeeze=True)

        if logscale:
            plt.yscale('log')

        axis[0].set_title('{} - std: {}'.format(titles[0], df1[column].std()))
        axis[1].set_title('{} - std: {}'.format(titles[1], df2[column].std()))
        axis[2].set_title('{} - std: {}'.format(titles[2], df3[column].std()))
        sns.histplot(df1[column][df1[column].notnull()],
                     kde=kde, color='b', ax=axis[0], stat=stat)
        sns.histplot(df2[column][df2[column].notnull()],
                     kde=kde, color='r', ax=axis[1], stat=stat)
        sns.histplot(df3[column][df3[column].notnull()],
                     kde=kde, color='g', ax=axis[2], stat=stat)


def compare_distributions_countplot_3by3(df1, df2, df3, col, titles=(['df1', 'df2', 'df3']), logscale=False):
    """
    Plot 3 count plots next to each other

    """

    # Padding
    rect = [0, 0.03, 1, 0.95]

    col.sort()

    available_columns = list(df1.columns)
    #     print(available_columns)

    for c in col:

        if c in available_columns:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 4))

            if logscale:
                plt.yscale('log')

            p1 = sns.countplot(x=c, data=df1, ax=ax1)
            p2 = sns.countplot(x=c, data=df2, ax=ax2)
            p3 = sns.countplot(x=c, data=df3, ax=ax3, hue='RESPONSE')

            ax1.set_title(titles[0])
            ax2.set_title(titles[1])
            ax3.set_title(titles[2])

            _ = plt.setp(p1.get_xticklabels(), rotation=90)
            _ = plt.setp(p2.get_xticklabels(), rotation=90)
            _ = plt.setp(p3.get_xticklabels(), rotation=90)

            fig.tight_layout(rect=rect)
            plt.show()


def plot_clusters(data, labels, label_color_map, title="First three PCA components coloured by cluster",
                  elev=30, azim=-130):
    """
    Plot the first 3 pca components colored by 8 clusters
    
    References:
    https://stackoverflow.com/questions/49564844/3d-pca-in-matplotlib-how-to-add-legend
    https://stackoverflow.com/questions/28227340/kmeans-scatter-plot-plot-different-colors-per-cluster
    """

    fig = plt.figure(figsize=(30, 30))
    ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)

    label_color = [label_color_map[l] for l in labels]

    # formulate the cluster label that will print inside the diagram. 
    # The label will appear near mean of the cluster
    labelTups = [('Cluster 0', 0), ('Cluster 1', 1), ('Cluster 2', 2), ('Cluster 3', 3),
                 ('Cluster 4', 4), ('Cluster 5', 5), ('Cluster 6', 6), ('Cluster 7', 7)]
    for name, label in labelTups:
        ax.text3D(data[labels == label, 0].mean(),
                  data[labels == label, 1].mean(),
                  data[labels == label, 2].mean(),
                  name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.7, edgecolor='w', facecolor='w'),
                  fontsize='xx-large')

    # construct the scatter plot
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=label_color,
               edgecolor='k',
               s=40,
               alpha=0.6)

    ax.set_title(title, {'fontsize': 40})
    ax.set_xlabel("PCA 1 - Middle aged - Top earner families", {'fontsize': 30})
    ax.set_ylabel("PCA 2 - Digital Media kids", {'fontsize': 30})
    ax.set_zlabel("PCA 3 - Older city lovers", {'fontsize': 30})
    ax.dist = 10

    fig.add_axes(ax)

    plt.show()


def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """

    # Create figure
    fig = pl.figure(figsize=(11, 5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(data[feature], bins=25, color='#00A0A0')
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(11, 8))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j // 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j // 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    #     # Add horizontal lines for naive predictors
    #     ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #     ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #     ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #     ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
              loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, x=0.63, y=1.05)
    # Tune the subplot layout
    # Refer - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html for more details on the arguments
    pl.subplots_adjust(left=0.125, right=1.2, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
    pl.tight_layout()
    pl.show()


def feature_importance_plot(importances, X_train, y_train):
    """ Display the five most important features"""
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    pl.bar(np.arange(5), values, width=0.6, align="center", color='#00A000', \
           label="Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', \
           label="Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc='upper center')
    pl.tight_layout()
    pl.show()


def plot_components(component, df, threshold=0.10):
    """ Plot features with weights within the threshold with a PCA component"""
    pca = df.loc[component]
    pca.loc[(pca > threshold) | (pca < -threshold)].sort_values(ascending=False).plot(kind="bar",
                                                                                      figsize=(15, 8),
                                                                                      title=component,
                                                                                      colormap='Blues_r')


def std_feature_importance(df1, df2, df_missing, top_n=50, title='Columns with biggest difference in std',
                           compare=['std', 'std_df2'], visualize=False):
    """ custom feature importance by comparing differences in std of a column in 2 datsets """

    columns_to_drop = list(df1.loc[:, df1.dtypes == 'object'].columns)
    df1.drop(columns_to_drop, inplace=True, axis=1)
    df2.drop(columns_to_drop, inplace=True, axis=1, errors='ignore')

    # scale the data before we calculate mean or std
    scaler = MinMaxScaler()
    df1_scaled = scaler.fit_transform(df1)
    df1 = pd.DataFrame(df1_scaled, columns=df1.columns, index=df1.index)
    df2_scaled = scaler.fit_transform(df2)
    df2 = pd.DataFrame(df2_scaled, columns=df2.columns, index=df2.index)

    # do not regard columns with missing values exceeded 70% as important
    if len(df_missing) > 0:
        columns_to_exclude = list(df_missing[df_missing >= 70].index)
        columns_to_exclude.append('RESPONSE')
        print('Columns to be excluded from features importance: {}'.format(columns_to_exclude))
        df1.drop(columns_to_exclude, axis=1, errors='ignore', inplace=True)
        df2.drop(columns_to_exclude, axis=1, errors='ignore', inplace=True)

        # calculate mean or std
    describe_df1 = df1.describe(include='all').transpose()
    describe_df2 = df2.describe(include='all').transpose()
    describe_joined = describe_df1.join(describe_df2, how='inner', rsuffix='_df2')

    describe_joined.head()
    df_all = describe_joined[compare].copy()

    df_all['diff'] = abs(df_all[compare[0]] - df_all[compare[1]])
    df_all = df_all.sort_values(by='diff', ascending=False)

    if visualize:
        df_all['diff'][:top_n].plot(kind='bar',
                                    figsize=(16, 6),
                                    title=title)
        plt.show()

    return list(df_all.index)


def xgb_feature_importance(xgb, bst):
    """ Plot feature importance for xgboost model """
    metric_list = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for metric in metric_list:
        plt.figure(figsize=(20, 15))
        xgb.plot_importance(bst, fmap=filename_model_featuremap,
                            ax=plt.gca(),
                            max_num_features=50,
                            show_values=True,
                            importance_type=metric,
                            title=('Feature Importance - ' + metric))
        plt.show()


def visualize_tree(xgb, bst, tree_to_plot=0):
    """ Visualize tree for XGBoost model 
    
    # gain: the average gain across all splits the feature is used in.
    # weight: the number of times a feature is used to split the data across all trees.
    # cover: the average coverage across all splits the feature is used in.
    # total_gain: the total gain across all splits the feature is used in.
    # total_cover: the total coverage across all splits the feature is used in.
    """

    tree_to_plot = tree_to_plot
    plot_tree(bst, fmap=filename_model_featuremap, num_trees=tree_to_plot, rankdir='LR')

    fig = plt.gcf()

    # Get current size
    fig_size = fig.get_size_inches()

    # Set zoom factor 
    sizefactor = 20

    # Modify the current size by the factor
    plt.gcf().set_size_inches(sizefactor * fig_size)

    # The plots can be hard to read (known issue). So, separately save it to a PNG, which makes for easier viewing.
    # fig.savefig('tree' + str(tree_to_plot)+'.png')
    plt.show()
