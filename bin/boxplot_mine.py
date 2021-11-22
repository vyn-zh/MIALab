import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class metrics:
    Dice = "DICE"
    HausdorffDistance = "HDRFDST"


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
    xLabels = ['']
    for label in np.unique(pd_data.LABEL):
        plotData.append(pd_data[pd_data.LABEL==label][metric])
        xLabels.append(label)
    return plotData, xLabels


#%%    
    

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot
    
    # results = readFile(False, '2021-11-22-09-31-45')
    
    OL_props = dict(markerfacecolor='g', marker='D', markersize=5)
    outliers_weight = 1.2  #default 1.5! (factor to multiply with IQR)
    
    #%% Plot DICE
    metric = metrics.Dice
    results = readFile('2021-11-22-09-31-45-HD95')
    plotData, xLabels = prepareData(results, metric)
    
    plt.figure('Boxplots', figsize=(7.2, 9.6))
    plt.subplot(311)
    r = plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
                    vert = 0)
    plt.grid(axis='y')
    plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
               fontsize=8)
    plt.title('{}'.format(metric), fontsize=9)
    
    print('***** Outliers for DICE:')

    
    #%% Plot HD 95
    metric = metrics.HausdorffDistance
    results = readFile('2021-11-22-09-31-45-HD95')
    plotData, xLabels = prepareData(results, metric)
    
    plt.subplot(312)
    r = plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
                    vert = 0)
    plt.grid(axis='y')
    plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
               fontsize=8)
    plt.title('{} 95'.format(metric), fontsize=9)
    
    print('***** Outliers for HD95:')

    
    
    #%% Plot HD 100
    results = readFile('2021-11-22-10-23-58-HD100')
    plotData, xLabels = prepareData(results, metric)
    
    plt.subplot(313)
    r = plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
                    vert = 0)
    plt.grid(axis='y')
    plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
               fontsize=8)
    plt.title('{} 100'.format(metric), fontsize=9)
    
    print('***** Outliers for HD100:')

    
    #%%
    
    plt.suptitle('Outliers @ {}*IQR'.format(outliers_weight),
                 fontweight='bold')
    plt.tight_layout()
    plt.show()


    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
if __name__ == '__main__':
    main()
