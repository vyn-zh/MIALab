import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


class metrics:
    Dice = "DICE"
    HausdorffDistance = "HDRFDST"
    SurfDice = "SURFDICE"


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
    # fname_DICE = '2021-12-05-11-02-42-HD100'
    # fname_HD95 = '2021-11-22-09-31-45-HD95'
    # fname_HD100 = '2021-12-05-11-02-42-HD100'
    
    OL_props = dict(markerfacecolor='g', marker='D', markersize=5)
    outliers_weight = 1  #default 1.5! (factor to multiply with IQR)
    
    #%% Plot DICE
    metric = metrics.Dice
    # results_DICE = readFile(fname_DICE)
    results_DICE = readFile()
    plotData, xLabels = prepareData(results_DICE, metric)
    
    plt.figure('Boxplots',figsize=(7.2, 9.6))
    plt.subplot(311)
    plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
                vert = 0)
    plt.grid(axis='y')
    plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
               fontsize=8)
    plt.title('{}'.format(metric), fontsize=9)
    plt.xlim(0,1)
    
    print('\n***** Outliers for DICE: *****\n')
    
    outliers_DICE = getOutliers(results_DICE, metric)
    print(outliers_DICE)

    
    #%% Plot HD 95
    metric = metrics.SurfDice
    results_SURFDICE = readFile()
    plotData, xLabels = prepareData(results_SURFDICE, metric)
    
    plt.subplot(312)
    plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
                vert = 0)
    plt.grid(axis='y')
    plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
                fontsize=8)
    plt.title('{}'.format(metric), fontsize=9)
    plt.xlim(0,1)
    
    print('\n***** Outliers for SURFDICE *****\n')
    outliers_HD95 = getOutliers(results_SURFDICE, metric)
    print(outliers_HD95)
    # metric = metrics.SurfDice
    # results_SURFDICE = readFile()
    # plotData, xLabels = prepareData(results_SURFDICE, metric)
    # plotData_prev, xLabels = prepareData(results_DICE, metrics.Dice)
    # plotData.extend(plotData_prev)    

    
    # plt.subplot(312)
    # plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
    #             vert = 0)
    # plt.grid(axis='y')
    # plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
    #            fontsize=8)
    # plt.title('{}'.format(metric), fontsize=9)
    # plt.xlim(0,1)
    
    # print('\n***** Outliers for SURFDICE *****\n')
    # outliers_HD95 = getOutliers(results_SURFDICE, metric)
    # print(outliers_HD95)
    
    
    #%% Plot HD 100
    results_HD100 = readFile()
    metric = metrics.HausdorffDistance
    plotData, xLabels = prepareData(results_HD100, metric)
    
    plt.subplot(313)
    plt.boxplot(plotData, flierprops=OL_props, whis=outliers_weight,
                vert = 0)
    plt.grid(axis='y')
    plt.yticks(ticks=np.arange(len(xLabels)), labels=xLabels, rotation=20,
               fontsize=8)
    plt.title('HD100[mm]', fontsize=9)
    
    print('\n***** Outliers for HD100: *****\n')
    outliers_HD100 = getOutliers(results_HD100, metric)
    print(outliers_HD100)
    
    
    #%%
    
    # plt.suptitle('Boxplot comparison'.format(outliers_weight),
    #              fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
    #%%
    '''
    outliers_DICE['flag']=1
    outliers_HD100['flag']=2
    flag = np.zeros(50)
    flag[outliers_DICE.index] = 1
    flag[outliers_HD100.index] = 2
    
    df_ggplot = pd.DataFrame(dict(HD=results_HD100.HDRFDST, DICE = 1-results_DICE.DICE, flag = flag));

    df_ggplot = pd.DataFrame(dict(HD=results_HD100.HDRFDST[results_HD100.LABEL=='Amygdala'],
                                  DICE = 1-results_DICE.DICE[results_HD100.LABEL=='Amygdala'],
                                  flag = flag[results_DICE[results_DICE.LABEL=='Amygdala'].index]));
    
    
    df_ggplot.plot.scatter('HD', 'DICE', c='flag', cmap='viridis')
    # sns.scatterplot(x='HD', y='DICE', data=df_ggplot)
    '''

#%%
    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
if __name__ == '__main__':
    main()
