import numpy  as np  
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import ttest_ind
from matplotlib.colors import LinearSegmentedColormap


def plot_bars(df: pd.DataFrame, features: list , n_rows: int, n_cols: int, sort = False, log = False):
    """
    From dataframe plot several bar plots for the features contained in the list.
    Plots are distributed in rows and columns
    """
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features, start=1):
        counts = df[feature].value_counts()
        if log:
            counts = np.log10(counts)
            ylabel = 'log(count)'
        else:
            ylabel = 'Count'
        if not sort:
            counts = counts.loc[sorted(counts.index)]
        plt.subplot(n_rows, n_cols, i)
        plt.title(feature)
        plt.ylabel(ylabel)
        plt.xlabel("")
        ax = counts.plot.bar()
        ax.set_xticklabels(ax.get_xticklabels(), 
                      rotation=45,
                      ha='right',  # Horizontal alignment
                      rotation_mode='anchor')
        plt.tight_layout()  # Prevent label clipping
    plt.show()

def plot_density(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
    """
    From dataframe plot several density plots for the features contained in the list.
    Plots are distributed in rows and columns
    """
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features,start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(df[feature], fill=True, color='skyblue', alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Density')
    plt.show()

def plot_density_cat(df: pd.DataFrame, features: list , target: str, n_rows: int, n_cols: int):
    """
    Plot kernel density estimates (KDE) for multiple features stratified by a categorical target.
    """
    num_features = len(features)
    plt.figure(figsize=(6 * n_cols , 4 * n_rows)) 
    for i, feature in enumerate(features,start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(data=df, x=feature, fill=True, hue=target, alpha=0.5, palette='viridis')
        sns.kdeplot(data=df, x=feature, linewidth=0.5, color='black')
        plt.xlabel(feature)
        plt.ylabel('Density')
    plt.show()
    
def plot_two_density_cat(df: pd.DataFrame, features: list , target: str):
    """
    Plot kernel density estimates (KDE) for multiple features stratified by a categorical target.
    
    Generates a grid of subplots showing KDE plots for each specified feature, with distributions
    separated by the target categories. Each feature appears in two adjacent subplots:
    - Left subplot: KDE with separate density curves (common_norm=False)
    - Right subplot: KDE with normalized density curves (common_norm=True)
    """
    num_features = len(features)
    plt.figure(figsize=(12, 4 * num_features))  # Dynamic figure size
    for i, feature in enumerate(features,start=1):
        plt.subplot(num_features, 2, i*2)
        sns.kdeplot(data=df, x=feature, fill=True, hue=target, alpha=0.5, palette='viridis')
        sns.kdeplot(data=df, x=feature, linewidth=0.5, color='black')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.subplot(num_features, 2, (i*2)-1)
        sns.kdeplot(data=df, x=feature, fill=True, hue=target,
            alpha=0.5, palette='viridis', common_norm = False)
        plt.xlabel(feature)
        plt.ylabel('Density')
    plt.show()
    
def plot_bars_target(df: pd.DataFrame, features: list , target: str, sort = False, log = False):
    """
    Generate a grid of bar plots showing target distribution in levels and frequency of levels.
    """
    n_cols = 2
    n_rows = len(features)
    fig, axes = plt.subplots(n_rows,n_cols,figsize=(4 * n_cols ,3 * n_rows))
    axes = axes.flatten()
    width = 0.5
    for ax1, ax2, feature in zip(axes[::2], axes[1::2], features):
        # count number of ocurrences
        counts = df[feature].value_counts()
        counts.index = list(map(str, counts.index.tolist()))
        # contingency table
        prop = pd.crosstab(df[feature],df[target], normalize = 'index')
        prop.index = list(map(str, prop.index.tolist()))
        # some modifications 
        if log:
            counts = np.log10(counts)
            ylabel = 'log(count)'
        else:
            ylabel = 'Count'
        if not sort:
            counts = counts.loc[sorted(counts.index)]
        # plot distribution of target in levels
        prop = prop.reindex(counts.index)
        levels = prop.index.astype(str).tolist()
        values = prop.to_dict('list')
        bottom = np.zeros(prop.shape[0])
        for boolean, value in values.items():
            ax1.bar(levels, value, width, label=boolean, bottom=bottom)
            bottom += value
        ax1.set_title(feature)
        ax1.legend(loc="upper right")
        # plot distribution of labels 
        levels = counts.index
        value = list(counts)
        ax2.bar(levels, value, width)
        ax2.set_title(feature)
    plt.tight_layout()  # Prevent label clipping
    plt.show()
    
def plot_estimator_feature_contquant(df: pd.DataFrame, estimator: str, features: list , n_rows: int, n_cols: int):
    """
    Scatter plots features vs estimator showing PCC and MI
    """
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features['feature'],start=1):
        plt.subplot(n_rows, n_cols, i)
        plt.subplots_adjust(hspace=0.4)
        plt.plot(df[feature], df[estimator], 'o')
        plt.xlabel(feature)
        plt.ylabel(estimator)
        plt.title("PCC={:.2f}, MI={:.2f}".format(features['pcc'][i-1], features['mi'][i-1]),
              fontsize=16)
    plt.show()


def plot_stat_vs_mi_num(features, label):
    """
    Scatter plot for statistic vs MI for numerical variables
    """
    plt.figure(figsize=(6, 5))
    plt.plot(features['stat'], features['mi'],'o', markersize = 5)
    texts = [plt.text(
        features['stat'][i], features['mi'][i], features['feature'][i], ha='center', va='center', fontsize = 10) 
             for i in range(len(features['feature']))]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'), force_text = 5)
    
    plt.xlabel("MI")
    plt.ylabel(label)
    plt.show()
    
def plot_stat_vs_mi_cat(df, label):
    """
    Scatter plot for statistic vs MI for categorical variables
    """
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x='stat', y='mi', hue='feature', palette='viridis', s=20)

    texts = [plt.text(df['stat'][i], df['mi'][i], df['level'][i], ha='center', va='center', fontsize = 10) for i in range(len(df['level']))]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'), force_text = 5)

    plt.xlabel("MI")
    plt.ylabel(label)
    plt.legend(title='Feature')
    plt.show()
    
def plot_estimator_feature_qualit_bi(df: pd.DataFrame, estimator: str, features: list , n_rows: int, n_cols: int):
    """
    Violin plots and t-test p-value for correlation between estimator and features with two factors
    """
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features,start=1):
        group1 = df[df[feature] == 0][estimator]
        group2 = df[df[feature] == 1][estimator]
        t_stat, p_value = ttest_ind(group1, group2)

        plt.subplot(n_rows, n_cols, i)
        plt.subplots_adjust(hspace=0.4)
        sns.violinplot(x=feature, y=estimator, data=df, inner='quartile', palette='pastel')
        plt.xlabel(feature)
        plt.ylabel(estimator)
        plt.title("t-Test p-value ={:.2E}".format(p_value),
              fontsize=16)
    plt.show()

def get_indexes_from_list(lst: list, targets: list):
    """
    Return the indexes in input list (lst) for the list of features in targets
    """
    return(list(filter(lambda x: lst[x] in targets, range(len(lst)))))

def get_elements_from_list(lst: list, indexes: list):
    """
    Return the elements of list lst in positions indexes
    """
    return(lst[i] for i in indexes)

def na_plot(df, threshold = 1):
    """
    Visualizes columns with missing values exceeding a specified threshold percentage.
    """
    df_na_features = df.isna().mean()*100
    df_na_features = df_na_features[df_na_features > threshold]
    fig_width = max(df_na_features.shape[0] * 0.3, 3)
    plt.figure(figsize=(fig_width,4))
    ax = df_na_features.sort_values(ascending=False).plot.bar()
    ax.set_xticklabels(ax.get_xticklabels(), 
                      rotation=45,
                      ha='right',  # Horizontal alignment
                      rotation_mode='anchor')
    plt.ylabel("Percentage of NA values")
    plt.tight_layout()
    plt.show()
    
def heatmap_triangle(df, label):
    """
    Plot a lower triangular heatmap with values between 0 and 1.
    """
    # generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 5))

    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8, "label": label})

    plt.show()
    
def heatmap_threshold(df, threshold, label):
    """
    Plot a heatmap with a threshold-based colormap, where values above the threshold
    are displayed in solid blue and values below use a gradient (YlGnBu).
    """
    nrows = df.shape[0]
    ncols = df.shape[1]
    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(ncols, nrows))
    # create custom colormap
    n_threshold_colors = int(threshold*256)
    cmap = plt.get_cmap('YlGnBu', n_threshold_colors)
    new_colors = cmap(np.linspace(0, 1, n_threshold_colors))
    # set colors above threshold to solid blue
    blue_color = np.array([0.03137255, 0.11372549, 0.34509804, 1])
    new_colors = np.vstack([new_colors, np.tile(blue_color,(256-n_threshold_colors,1))])
    new_cmap = LinearSegmentedColormap.from_list('trunc_YlGnBu', new_colors)

    sns.heatmap(df, cmap=new_cmap, vmin=0, vmax=1,
                center=0.5, linewidths=.1, 
                annot = True, fmt = ".2f",
                cbar_kws={"shrink": .8, "label": label})
    
    plt.show()
