# -*- coding: utf-8 -*-
"""
@Ed Morrell

Wave morph function
- Functions to look at various measures of ECG and PPG wave morphology
"""
import heartpy as hp
from biosppy.signals import ecg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
from scipy import signal
from scipy.ndimage import gaussian_filter

def mw_conv(sig,N=100):
    '''
    Convolve signal with moving window to remove high freq. noise and produce moving-window average 
    Input:
          - Signal to convolve
          - N: Size of moving window     
    Output:
          - Convolved signal
    '''
    #Moving window average
    return np.convolve(sig, np.ones(N)/N, mode='valid')


def av_wf(ppg_sig,ppg_ts,ppg_pts):
    '''
    Produces an array of normalized time-invariant waveform
    
    Inputs:
            - ppg_sig: PPG signal
            - ppg_ts: PPG timestamps
            - ppp_pts: PPG peak timestamps
    
    Outputs:
            - wfs: Array of every waveform
    '''  
    
    #Nested scale function
    def scale(im, nC):
        nC0 = len(im)     # source number of vals
        return [[im[int(nC0 * c / nC)]]  for c in range(nC)]
    
    #Computes each time invariant waveform as mid-mid point
    wfs = []
    for i in range(0,len(ppg_pts)-2):
        
        #Find indices of 1st 2 peaks
        start_ind = np.where(ppg_ts == ppg_pts[i])[0][0]
        end_ind = np.where(ppg_ts == ppg_pts[i+1])[0][0]
        
        #Get midpoint as start of wave
        w_start = round(((end_ind - start_ind) / 2) + start_ind)
        
        #Next 2 peaks
        start_ind = np.where(ppg_ts == ppg_pts[i+1])[0][0]
        end_ind = np.where(ppg_ts == ppg_pts[i+2])[0][0]

        #Midpoint as end of wave
        w_end = round(((end_ind - start_ind) / 2) + start_ind)
        
        #Sample wave
        f = ppg_sig[w_start:w_end]

        #Normalize f
        f = (f-np.min(f))/(np.max(f)-np.min(f))
    
        #Scale f so all waves same length
        f = scale(f,50)
        
        wfs.append(f)
    
    wfs = np.array(wfs)[:,:,0]
    
    return wfs


def plot_av_wf(wfs,p_color='k'):
    '''
    Takes waveforms array and plots (plot STD rather than SEM as SEM too small)
    '''
    
    mean_wf = wfs.mean(axis=0)
    sem_wf = wfs.std(axis=0)
    
    fig = plt.figure()
    x = np.linspace(0,50,50)
    plt.plot(x,mean_wf,p_color)
    plt.fill_between(x,mean_wf-sem_wf, mean_wf+sem_wf,color=p_color,alpha=0.7)
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def STT(ppg_sig,ppg_ts,peaks):
    '''
    Generates array of STTs from PPG signal
    
    Inputs:
           - ppg_sig: Raw ppg signal
           - ppg_ts: Timestamps
           - peaks: Peaks dataframe
           
    Output:
           - STT: Array of Slope-Traversal Times
    '''
    #Compute height threshold (2/3 of max peak)
    h_th = -(2/3*np.mean(peaks['peak_value']))
    
    #Find negative peak positions
    neg_peaks = signal.find_peaks(-ppg_sig,height=h_th)
    
    #Finds negative peak times
    neg_pt = ppg_ts[neg_peaks[0]]
    
    #Computes time from neg peak to primary peak
    STT_array = np.zeros(len(peaks)-1)
    for peak in range(0,len(peaks['peak_time'])-1):
        np_ind = np.where((neg_pt < peaks['peak_time'].iloc[peak+1]) 
                          & (neg_pt > peaks['peak_time'].iloc[peak]))
        if len(np_ind[0]) == 0:
            STT_array[peak] = np.nan
        else:
            p_start = neg_pt[max(np_ind[0])] #Takes max value in case multiple peaks falling in range
            p_end =  peaks['peak_time'].iloc[peak+1]

            STT_array[peak] = p_end - p_start #Adds difference to array as STT
    
    return STT_array

def MW_PAT_Estimate(ecg,peaks,sig_ts):
    '''
    Noisy estimate of PAT from ECG signal and PPG signal using moving window method
        - PAT measures time from peak of ECG signal to peak of PPG signal as a metric for time
          taken for pulse to travel from heart to PPG recording site
    
    Input:
          - ecg: ECG signal
          - peaks: Peaks dataframe
          - sig_ts: Numpy array of timestamps
          
    Output:
           - PAT_array: Pulse-Arrival Time Estimate
    '''
    
    #Moving window average
    N = 30 #Size of window
    ecg_mwa = np.convolve(ecg, np.ones(N)/N, mode='valid') #Convolved signal
    
    #Compute the height threshold as at least 1SD > than mean of the signal
    h_th = np.average(ecg_mwa) + np.std(ecg_mwa)
    
    #Compute ECG peaks
    ecg_peaks = signal.find_peaks(ecg_mwa,height=h_th)
    
    #Finds ECG peak times
    ecg_pt = sig_ts[ecg_peaks[0]]
    
    #Computes time from ECG peak to PPG peak
    PAT_array = np.zeros(len(peaks)-1)
    for peak in range(0,len(peaks['peak_time'])-1):
        
        #Query range only accepts peaking falling in second half of interval between PPG peaks
        q_range = (peaks['peak_time'].iloc[peak+1] - peaks['peak_time'].iloc[peak]) / 2
        
        #Query times to check for peaks
        q_times = [(peaks['peak_time'].iloc[peak+1] - q_range),peaks['peak_time'].iloc[peak+1]]
        
        pt_ind = np.where((ecg_pt < q_times[1]) 
                          & (ecg_pt > q_times[0]))
        
        if len(pt_ind[0]) == 0:
            PAT_array[peak] = np.nan
        else:
            #Takes the peak value with largest signal value
            e_peak = ecg_pt[pt_ind[0][np.argmax(ecg_mwa[ecg_peaks[0][pt_ind[0]]])]]

            p_peak =  peaks['peak_time'].iloc[peak+1]

            PAT_array[peak] = p_peak - e_peak #Adds difference to array as STT
    
    return PAT_array


def ppg_slope(ppg_sig,peaks,sig_ts):
    '''
    Computes peak slope (max gradient) of each PPG signal
    
    Inputs:
           - ppg_sig: ECG signal
           - peaks: Peaks dataframe
           - sig_ts: Numpy array of timestamps
    
    Output:
           - slope_array: Array of slope values
    '''
    #Compute derviative of signal
    ppg_d = np.gradient(ppg_sig)
    
    #Compute peaks (>2SD from mean of signal)
    d_peaks = signal.find_peaks(ppg_d,(2*np.std(ppg_d)) + np.mean(ppg_d))
    
    #Finds negative peak times
    d_pt = sig_ts[d_peaks[0]]
    
    #Computes time from neg peak to primary peak
    slope_array = np.zeros(len(peaks)-1)
    for peak in range(0,len(peaks['peak_time'])-1):
        p_ind = np.where((d_pt < peaks['peak_time'].iloc[peak+1]) 
                          & (d_pt > peaks['peak_time'].iloc[peak]))
        if len(p_ind[0]) == 0:
            slope_array[peak] = np.nan
        else:
            #Takes the peals
            slope_array[peak] = max(ppg_sig[d_peaks[0][p_ind]])
    
    return slope_array


def P1_Amp(ppg_sig,ppg_ts,peaks):
    '''
    Generates array of P1 amplitudes (distance from base to height of primary peak)
    
    Inputs:
           - ppg_sig: Raw ppg signal
           - ppg_ts: Timestamps
           - peaks: Peaks dataframe
           
    Output:
           - P1_Array: Array of Slope-Traversal Times
    '''
    #Compute height threshold (2/3 of max peak)
    h_th = -(2/3*np.mean(peaks['peak_value']))
    
    #Find negative peak positions
    neg_peaks = signal.find_peaks(-ppg_sig,height=h_th)
    
    #Finds negative peak times
    neg_pt = ppg_ts[neg_peaks[0]]
    
    #Computes time from neg peak to primary peak
    P1_array = np.zeros(len(peaks)-1)
    for peak in range(0,len(peaks['peak_time'])-1):
        np_ind = np.where((neg_pt < peaks['peak_time'].iloc[peak+1]) 
                          & (neg_pt > peaks['peak_time'].iloc[peak]))
        if len(np_ind[0]) == 0:
            P1_array[peak] = np.nan
        else:
            
            w_peak = peaks['peak_value'].iloc[peak+1] #Amplitude at waveform peak
            
            w_base = ppg_sig[neg_peaks[0][max(np_ind[0])]] #Amplitude at waveform base
            
            P1_array[peak] = w_peak - w_base #Adds amp to array
    
    return P1_array


def d_notch(ppg_sig,peaks,ppg_ts):
    '''
    Computes time from main peak to dicrotic notch
    
    Inputs:
           - ppg_sig: Raw ppg signal
           - peaks: Peaks dataframe
           - ppg_ts: Timestamps
           
    Output:
           - P1_Array: Array of Slope-Traversal Times
    '''
    
    #Filters signal usin chebyshev filter to try and emphasize dicrotic notch
    sos = signal.cheby2(10, 5, [0.5,5], 'bp', fs=250, output='sos')
    filtered = signal.sosfilt(sos, ppg_sig)
    
    #Finds filtered signal peaks
    f_peaks = signal.find_peaks(filtered)
    
    #Finds negative peak times
    f_pt = ppg_ts[f_peaks[0]]
    
    #Takes first peak appearing > 50ms after PPG peak as dichrotic notch and 
    #computes time from PPG peak to DC
    dn_array = np.zeros(len(peaks)-1)
    for peak in range(0,len(peaks['peak_time'])-1):
        
        #Query times to check for peaks
        q_times = [(peaks['peak_time'].iloc[peak] + 50000),peaks['peak_time'].iloc[peak+1]]
        
        #Find indices of all peaks falling between 2 ppg peaks and more than 50ms after the first
        p_ind = np.where((f_pt > q_times[0]) & (f_pt < q_times[1]))
        
        if len(p_ind[0]) == 0:
            dn_array[peak] = np.nan
        else:
            
            p_time = peaks['peak_time'].iloc[peak] #Time of PPG peak
            
            dc_notch = f_pt[p_ind[0][0]] #Takes first index as time of notch
            
            dn_array[peak] = dc_notch - p_time #Adds time between peak and notch to array
    
    return dn_array