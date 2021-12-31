import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


class metrics:
    Dice = "DICE"
    HausdorffDistance = "HDRFDST"
    SurfaceDice = "SURFDICE"


def readFile(fname=''):
    if len(fname) == 0:
        #automatic selection of most recent results
        tmp = os.listdir('mia-result/')
        tmp.sort()
        fpath = 'mia-result/{}/results.csv'.format(tmp[-1])
    else:
        #manual selection of desired timestamp
        fpath = 'mia-result/{}/results.csv'.format(fname)
    
    results = pd.read_csv(fpath, delimiter=';')
    return results


def prepareData(pd_data, metric):
    plotData = []
    xLabels = []
    for label in np.unique(pd_data.LABEL):
        plotData.append(pd_data[pd_data.LABEL==label][metric])
        xLabels.append(label)
    return plotData, xLabels


def getOutliers(pd_data, metric):
    df = pd.DataFrame()
    for label in np.unique(pd_data.LABEL):
        data = pd_data[pd_data.LABEL==label][metric]
        q1 = np.quantile(data, 0.25)
        # finding the 3rd quartile
        q3 = np.quantile(data, 0.75)
        # finding the iqr region
        iqr = q3-q1
        # finding upper and lower whiskers
        factor = 1
        upper_bound = q3+(factor*iqr)
        lower_bound = q1-(factor*iqr)
        if(metric == metrics.HausdorffDistance):
            for i in data[(data >= upper_bound)].index:
                df = df.append(pd_data[pd_data.index == i])
        else:
            for i in data[(data <= lower_bound)].index:
                df = df.append(pd_data[pd_data.index == i])
    return df


#%%    
    

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot
    
    # results = readFile('2021-11-22-09-31-45')
    # fname = '2021-12-05-11-02-42-HD100'
    fname = ''
    results = readFile(fname)
    
    #%% save results for different metrics

    diceData, labels = prepareData(results, metrics.Dice)
    hdData, _ = prepareData(results, metrics.HausdorffDistance)
    surfDiceData, _ = prepareData(results, metrics.SurfaceDice)
    
    #%% get outliers for each metric
    outliers_DICE = getOutliers(results, metrics.Dice)
    outliers_HD = getOutliers(results, metrics.HausdorffDistance)
    outliers_SURFDICE = getOutliers(results, metrics.SurfaceDice)
    
    #%% #%% Perform comparison HD vs. Surface DICE
    N = results.shape[0]
    flag = np.zeros(N, dtype=int)
    flag_ = np.zeros(N, dtype=int)

    # set outliers for DICE = 1
    flag[outliers_DICE.index] = 1
    flag_[outliers_DICE.index] = 1
    
    # set outliers HD = 2
    flag[outliers_HD.index] = 2
    # set both outliers = 3
    # flag[np.where(np.logical_and(flag==2, flag_))[0]]=3
    
    # generate labels for 0-1-2-3
    flag_labels = ['general inlier', 'DICE outlier', 'HD100 outlier']
    colors = ['#00008B', '#FF00FF','#228B22']
    markers = ['^', '1', 'x', '+', 'o']
    marker_fill = [False,True, True, True, False]
    
    
    scatter_DICE_HD = pd.DataFrame(dict(HD=results.HDRFDST,
                                        DICE = 1-results.DICE,
                                        flag = flag));
    
    fig = plt.figure('DICE vs HD100')
    ax_g = fig.add_subplot(111)
    
    
    for i in range(len(labels)):
        for f in np.unique(flag):
            plt.scatter(
                scatter_DICE_HD.HD[results.LABEL==labels[i]][scatter_DICE_HD.flag==f],
                scatter_DICE_HD.DICE[results.LABEL==labels[i]][scatter_DICE_HD.flag==f],
                edgecolor=colors[f], c=colors[f] if marker_fill[i] else "none",
                s=50, marker=markers[i], label=flag, linewidth=1.3)
    ax_g.set_xlabel('HD100[mm]')
    ax_g.set_ylabel('1-DICE')
    ax_g.set_xlim(0)
    ax_g.set_ylim(0,1)
    ax_g.set_aspect(max(scatter_DICE_HD.HD))
    ax_g.grid()
    ax_g.set_title('DICE vs HD100')
    
    legend_colors=[]
    for i in range(len(colors)):
        legend_colors.append(mpatches.Patch(color=colors[i],
                                            label=flag_labels[i]))

    legend_markers=[]
    for i in range(len(markers)):
        legend_markers.append(mlines.Line2D([], [], marker=markers[i],
                          markersize=6, label=labels[i], ls="",
                          markeredgecolor='black',
                          c='black' if marker_fill[i] else "none"))
        
    l1 = ax_g.legend(handles=legend_colors,
                bbox_to_anchor=(0.0,1), loc="upper left",
                ncol=1, prop={'size': 8})
    ax_g.legend(handles=legend_markers,
                bbox_to_anchor=(1.,1), loc="upper right",
                ncol=1, prop={'size': 8})
    fig.add_artist(l1)
    
    plt.tight_layout()
    
    flag[outliers_SURFDICE.index] = 1
    flag_[outliers_SURFDICE.index] = 1
    # set both outliers = 3
    # flag[np.where(np.logical_and(flag==2, flag_))[0]]=3
        
    scatter_SURFDICE_HD = pd.DataFrame(dict(HD=results.HDRFDST,
                                            SURFDICE = 1-results.SURFDICE,
                                            flag = flag));
    
    fig = plt.figure('SURFDICE vs HD100')
    ax_g = fig.add_subplot(111)
    for i in range(len(labels)):
        for f in np.unique(flag):
            plt.scatter(
                scatter_SURFDICE_HD.HD[results.LABEL==labels[i]][scatter_SURFDICE_HD.flag==f],
                scatter_SURFDICE_HD.SURFDICE[results.LABEL==labels[i]][scatter_SURFDICE_HD.flag==f],
                edgecolor=colors[f], c=colors[f] if marker_fill[i] else "none",
                s=50, marker=markers[i], label=flag, linewidth=1.3)
    ax_g.set_xlabel('HD100[mm]')
    ax_g.set_ylabel('1-SURFDICE')
    ax_g.set_xlim(0)
    ax_g.set_ylim(0,1)
    ax_g.set_aspect(max(scatter_SURFDICE_HD.HD))
    ax_g.grid()
    ax_g.set_title('SURFDICE vs HD100')

    
    l1 = ax_g.legend(handles=legend_colors,
                bbox_to_anchor=(0,1), loc="upper left",
                ncol=1, prop={'size': 8})
    ax_g.legend(handles=legend_markers,
                bbox_to_anchor=(1,1), loc="upper right",
                ncol=1, prop={'size': 8})
    fig.add_artist(l1)

    plt.tight_layout()
    
    

#%%
    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
if __name__ == '__main__':
    main()
