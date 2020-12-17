# -*- coding: utf-8 -*-
"""
@Ed Morrell

Heart-Rate Variability Analysis Functions
- Functions to assess heart-rate variabiltiy
"""
import heartpy as hp
from biosppy.signals import ecg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
from scipy.ndimage import gaussian_filter

def get_ppg_peaks(fn,fs=500):
    '''
    Produces a dataframe with peak_time and peak_value columns
    
    Inputs:
           - fn: Filename of .csv file
           - fs: Sampling freq. (Default: 500Hz)
                      
    Output: 
           - peaks: Pandas dataframe
    '''
    
    #load data from .csv file
    ppg_sig = hp.get_data(fn, column_name = 'ppg')

    #Process ppg data
    wd, m = hp.process(ppg_sig,fs)
    
    #Read data with pandas
    oDF = pd.read_csv(fn)
    ts = oDF['time']
    
    #Get peak times and values from peak indices
    peak_times = ts[wd['peaklist']]
    peak_vals = ppg_sig[wd['peaklist']]
    
    return pd.DataFrame({'peak_time':peak_times,'peak_value':peak_vals})


def plot_peaks(ts,signal,peaks,idx_range):
    '''
    Plots a signal with peaks
    
    Input:
          - ts: Signal timestamps
          - signal: Signal array
          - peaks: Peaks
          - idx_range: Tuple contatining range of indices to plot
    '''

    plt.plot(ts[idx_range[0]:idx_range[1]],signal[idx_range[0]:idx_range[1]])

    peak_inds = peaks[((peaks > idx_range[0]) & (peaks < idx_range[1]))]

    plt.plot(ts[peak_inds],signal[peak_inds],marker='o',ls='')

   
def get_NN_stats(ppg_sig,fs):
    '''
    Runs heartpy function to produce key NN stats
    
    Inputs:
            - ppg_sig: Numpy array of ppg signal
           
    Outputs:
            Dictionary of key NN stats
            - hr: Heart-rate (bpm)
            - ibi: Interburst interval (ms)
            - sdnn: NN SD (ms)
            - rmssd: Root-mean square rrow
            - pnn50: Proportion of NN intervals above 50ms
    '''
    _, measures = hp.process(ppg_sig,fs)
    return {'hr':measures['bpm'],'ibi':measures['ibi'],
            'sdnn':measures['sdnn'],'rmssd':measures['rmssd'],
            'pnn50':measures['pnn50']}


def full_session_stats():
    '''
    Creates dataframe of stats for each csv file in path
    
    Adds Hypertension column: 1 - Hypertension, 0 - Normotension, -1 - Hypotension
    
    Output: 
            - NN: Dataframe of NN stats for all patients included in fn_list
    '''
    
    nn_list = []
    class_vals = []
    
    #Goes through each csv file in path and computes NN stats (assumes class info in filename)
    for file in glob.glob('*.csv'):
        
        if 'hyper' in file.lower():
            f_class = 1
        elif 'norm' in file.lower():
            f_class = 0
        elif 'hypo' in file.lower():
            f_class = -1
        else:
            f_class = np.nan
        
        #Load dataframe
        df = pd.read_csv(file)
        
        #Extract ppg signal
        ppg = np.array(df['ppg'])
        
        if len(ppg)<600000:
            fs = 250
        else:
            fs = 500
        
        #Produce NN stats dict
        nns = get_NN_stats(ppg,fs)
        
        #Add stats and class to list
        nn_list.append(nns)
        class_vals.append(f_class)
    
    #Produce stats dataframe
    NN = pd.DataFrame(nn_list)
    
    #Add hypertension column
    NN['Hypertension'] = class_vals
    
    return NN


def compare_stats(NN):
    '''
    Produce a bar chart comparing key stats in 
    '''
    
    cols = NN.columns
    cols = cols.drop('Hypertension')
    
    plt.figure(figsize=(18,6))
    for i,col in enumerate(cols):
        means = np.array(NN[(NN['Hypertension'] == 1) | (NN['Hypertension'] == 0)].
                 groupby('Hypertension')[col].mean())
        stds = np.array(NN[(NN['Hypertension'] == 1) | (NN['Hypertension'] == 0)].
                 groupby('Hypertension')[col].std())
        
        sems = [val / np.sqrt(len(NN[NN['Hypertension']==i])) for i,val in enumerate(stds)]
        
        ax = plt.subplot(1,5,i+1)
        plt.bar([0,1],means,yerr=sems,tick_label=['Normotension','Hypertension'],
                color=['black','red'],alpha=0.8)
        plt.title(col)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        
def moving_window_func(df,win,step,fs=500):
    '''
    Moving window function to create sub arrays from ppf signal in df
    
    Inputs:
           - df: Dataframe containing ts and ppg signal
           - win: Size of moving window (minutes)
           - ol: Step-size of window function (minutes)
           
    Output:
           numpy array with window sized arrays
    '''
    #Convert window and overlap to samples
    win = int((win * 60)/(1/fs))
    step = int((step*60)/(1/fs))
    
    w_list = list()
    n_records = len(df)
    remainder = (n_records - win) % step 
    num_windows = 1 + int((n_records - win - remainder) / step)
    for k in range(num_windows):
        w_list.append(df['ppg'].iloc[k*step:win-1+k*step+1])
    return np.array(w_list)


def HRV_v_time(df,win,step,fs):
    '''
    Computes NN stats within moving windows to generate stats across time
    
    Inputs:
           - df: Dataframe containing ts and ppg signal
           - win: Size of moving window (minutes)
           - ol: Step-size of window function (minutes)
           
    Output:
           {Stat: value at each timepoint}       
    '''
    
    #Runs moving window function to get sub arrays
    arrays = moving_window_func(df,win,step,fs)
    
    #Creates empty array to add values to
    NN_arrs = np.zeros((5,len(arrays)))
    NN_keys  = []
    for i,arr in enumerate(arrays):
        stats = get_NN_stats(arr,fs)
        
        #Adds each stat to relevant numpy array
        ikey = 0
        for key,val in stats.items():
            NN_keys.append(key)
            NN_arrs[ikey][i] = val
            ikey += 1
    
    #Merge each array into dictionary
    return dict(zip(NN_keys,NN_arrs))


def hypertension_v_time(win,step):
    '''
    Generates NN stats vs. time for all files
    
    Adds Hypertension column: 1 - Hypertension, 0 - Normotension, -1 - Hypotension
    
    '''
    
    #Arrays to be saved to list of tuples
    hr = []
    ibi = []
    sdnn = []
    rmssd = []
    pnn50 = []
    
    #Goes through each csv file in path and computes NN stats (assumes class info in filename)
    for file in glob.glob('*.csv'):
        
        if 'hyper' in file.lower():
            f_class = 1
        elif 'norm' in file.lower():
            f_class = 0
        elif 'hypo' in file.lower():
            f_class = -1
        else:
            f_class = np.nan
        
        #Load dataframe
        df = pd.read_csv(file)
        
        if len(df)<600000:
            fs = 250
        else:
            fs = 500
        
        #Extract ppg signal
        NN_stats = HRV_v_time(df,win,step,fs)
        
        for key,array in NN_stats.items():
            eval('{}.append((f_class,array))'.format(key))
    
    return hr, ibi, sdnn, rmssd, pnn50


def plot_stat_v_time(stat_array,stat_str,step,smooth=False,sig=1):
    '''
    Plot stat_v_time array generated by hypertension v time
    
    Inputs:
           - stat_array: Arrays of stats vs. time generated using hypertension_v_time
           - stat_str: String with name of stat for plotting purposes
           - step: Step-size used to generate stat
           - smooth: Boolean indicating whether to smooth array (default: False)
           - sig: Sigma for use with gaussian smoothing function         
    '''
    
    #Remove hypotensive (can add in later if want to use)
    stat_array = [i for i in stat_array if i[0] != -1]
    
    #Find length of shortest segment
    min_len = min(len(i[1]) for i in stat_array)
    
    #Create an empty hypertensive and normotensive array: 1st row is mean value, 2nd row is SEM
    mean_hyp = np.zeros((2,min_len))  
    mean_norm = np.zeros((2,min_len))
    
    num_hyp = len([i for i in stat_array if i[0] == 0])
    num_norm = len([i for i in stat_array if i[0] == 1])
    
    norm = np.zeros((num_hyp,min_len))
    hyp = np.zeros((num_norm,min_len))
    i_norm = 0
    i_hyp = 0
    for arr in stat_array:
        if arr[0] == 0:
            norm[i_norm,:] = arr[1][0:min_len]
            i_norm += 1
        if arr[0] == 1:
            hyp[i_hyp,:] = arr[1][0:min_len]
            i_hyp += 1
    
    #Caclculate means
    mean_hyp[0,:] = np.matrix(hyp).mean(0)
    mean_norm[0,:] = np.matrix(norm).mean(0)
    
    #Calculate SEM
    mean_hyp[1,:] = np.matrix(hyp).std(0) / np.sqrt(len(np.matrix(hyp)))
    mean_norm[1,:] = np.matrix(norm).std(0) / np.sqrt(len(np.matrix(norm)))
    
    if smooth:
        mean_hyp[0,:] = gaussian_filter(mean_hyp[0,:], sigma=sig)
        mean_norm[0,:] = gaussian_filter(mean_norm[0,:], sigma=sig)
    
    fig = plt.figure(figsize=(6,4))
    x = np.arange(1,len(norm[0])+1)*step
    plt.plot(x,mean_norm[0],color='black',label='Normotensive')
    plt.fill_between(x, mean_norm[0]-mean_norm[1], mean_norm[0]+mean_norm[1],color='black',alpha=0.7)
    plt.plot(x,mean_hyp[0],color='red',label='Hypertensive')
    plt.fill_between(x, mean_hyp[0]-mean_hyp[1], mean_hyp[0]+mean_hyp[1],color='red',alpha=0.7)
    plt.xlabel('Time (mins)')
    plt.ylabel(stat_str)
    plt.legend(loc='center right')
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return mean_hyp, mean_norm
          

def meas_change(hyp,norm,step,meas_str,c_win=5):
    '''
    Compute and plot change from start to end of session
    
    Inputs:
           - hyp/norm: Mean_v_time array
           - step: Size of overlap
           - meas_str: Name of measure
           - c_win: Compute window (mins) - ie 5 would comput difference between first and last 5 minutes
    '''
    #Number of windows to sum
    wins = int(c_win/step)
    
    #Compute differences
    h_start = np.mean(hyp[0,0:wins])
    h_end = np.mean(hyp[0,-wins:])
    
    n_start = np.mean(norm[0,0:wins])
    n_end = np.mean(norm[0,-wins])
    
    #Compute mean SEM differences
    hs_start = np.mean(hyp[1,0:wins])
    hs_end = np.mean(hyp[1,-wins:])
    
    ns_start = np.mean(norm[1,0:wins])
    ns_end = np.mean(norm[1,-wins])
    
    means = [(n_end-n_start),(h_end-h_start)]
    sems = [(ns_end-ns_start),(hs_end-hs_start)]
    
    #Plot
    plt.bar([0,1],means,yerr=sems,tick_label=['Normotension','Hypertension'],
                color=['black','red'],alpha=0.8)
    plt.ylabel(meas_str)          