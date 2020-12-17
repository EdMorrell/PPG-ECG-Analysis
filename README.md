# PPG-ECG-Analysis
A range of functions and notebooks used to extract meaningful insights from PPG and ECG data 

## What insights can we gauge from PPG and ECG signals alone?
In these notebooks I attempt to extract data on:
1. ECG and PPG waveform morphology
2. Heart rate variability
3. Cardiovascular dynamics

## Functions:
1. **wavemorph.py**: Function for assessing waveform morphology
2. **hrv.py**: Functions to desrcibe and visualise heart rate variability
3. **time_embed.py**: Functions to produce time-delay embedding of signal and take Poincare sections of embedding

## Notebooks:
1. **Wave Morphology.ipynb**: Analysis of waveform morphology (ie. peaks, timing, waveform slope, amplitude etc.)
2. **Heart Rate Variability.ipynb**: Analysis of heart rate variability (ie. IBI, SDNN etc.)
3. **Time Delay Embedding.ipynb**: Analysis and visualisation using time-delay embedding

## Dependencies:
* *HeartPy*: Available [here](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/)
* *BioSPPY*: Available [here](https://biosppy.readthedocs.io/en/stable/)
* *Numpy*
* *Scipy*
* *Matplotlib*
* *Pandas*
* *Seaborn*
