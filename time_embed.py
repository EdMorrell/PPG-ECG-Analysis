# -*- coding: utf-8 -*-
"""
@Ed Morrell

Time-embedding functions
	- Produce time embedding of a signal and take Poincare sections of signal
"""
import numpy as np
import scipy as sp
import heartpy as hp
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from biosppy.signals import ecg
import rr
from mpl_toolkits import mplot3d
from scipy import signal

def zRotation(theta):
    """
    Rotation matrix about z-axis
    Input:
    theta: Rotation angle (radians)
    Output:
    Rz: Rotation matrix about z-axis
    """
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]], float)
    return Rz


def get_plane_vec(theta):
    '''
    Supply theta to get hyperplane point and plane normal vector
    Inputs:
            - theta: Angle of hyperplane (radians)
    Outputs:
            - sspTemplate: State-space template (point on hyperplane)
            - nTemplate: Plane normal (Vector orthogonal to hyperplane)
    '''
    # Set the angle between the Poincare section hyperplane and the x-axis
    thetaPoincare = theta

    # Define vectors which will be on and orthogonal to the Poincare section
    # hyperplane
    e_x = np.array([1, 0, 0], float)  # Unit vector in x-direction
    # Template vector to define the Poincare section hyperplane
    sspTemplate = np.dot(zRotation(thetaPoincare), e_x)  # Matrix multiplication

    # Normal to this plane will be equal to template vector rotated pi/2 about
    # the z axis
    nTemplate = np.dot(zRotation(np.pi / 2), sspTemplate)
    
    return sspTemplate, nTemplate


# Define the Poincare section hyperplane equation
def UPoincare(ssp, sspTemplate, nTemplate):
    """
    Plane equation for the Poincare section hyperplane which includes z-axis
    and makes an angle theta with the x-axis see ChaosBook ver. 14, fig. 3.2
    Inputs:
    ssp: State space point at which the Poincare hyperplane equation will be
         evaluated
    Outputs:
    U: Hyperplane equation which should be satisfied on the Poincare section
       U = (ssp - sspTemplate) . nTemplate (see ChaosBook ver. 14, eq. 3.6)
    """
    U = np.dot(ssp - sspTemplate, nTemplate)
    return U


def delay_embed(sig,fs,start=0,stop=50000):
    '''
    Produce time-delay embedding whereby X is signal, Y is signal + tau, 
    Z is signal + 2tau
        - Tau estimated as 1/4 of predominant component of power spec
        - X and Y normalized between -1 and 1 then centred by their means
        - Z normalized between 0 and 1  
        
    Inputs:
            - sig: Signal to embed
            - fs: Sampling frequency of signal
            - start: Start index
            - stop: End index (Default: 50000 samples)
    Outputs:
            - sspSolution: State-space solution (3d vector giving co-ordinates at each timepoint)
    '''

    #Produces a spectrogram
    f, Pxx_spec = signal.periodogram(sig,fs)

    #Finds predominant peak from spectrogram
    pp = f[np.argmax(Pxx_spec[0:3000])]

    tau = int(((1/pp)/4)*fs) #Uses 1/4 of period of predominant peak as time lag
    
    #Produces lag signals
    t0 = sig[start:stop]
    t1 = sig[start+tau:stop+tau]
    t2 = sig[start+2*tau:stop+2*tau]

    #Normalize signals
    t0 = 2*((t0 - np.min(t0)) / (np.max(t0)-np.min(t0)))-1
    t1 = 2*((t1 - np.min(t1)) / (np.max(t1)-np.min(t1)))-1
    t2 = (t2 - np.min(t2)) / (np.max(t2)-np.min(t0))

    #Centre signals
    t0 = t0 - np.mean(t0)
    t1 = t1 - np.mean(t1)
    t2 = t2 - np.mean(t2)
    
    #Creates 3-dimensional timepoint vector
    sspSolution = np.zeros((len(t0),3))
    sspSolution[:,0] = t0
    sspSolution[:,1] = t1
    sspSolution[:,2] = t2
    
    return sspSolution


def plot_embed(sspSolution,plot_intersect=False,sspTemplate=[],nTemplate=[]):
    '''
    Plot time-delay embedding
    
    Inputs:
           - sspSolution: 3d delay embedding vector
           - plot_intersect: boolean which if true plots points that intersect
                             hyperplane provided by sspTemplate
           - sspTemplate: vector of point on hyperplane
           - nTemplate: hyperplane vector normal
    '''
    
    plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    ax.plot(sspSolution[:,0],sspSolution[:,1],sspSolution[:,2],
           color='gray',linewidth=0.5)
    
    #Plot points intersecting hyperplane
    if plot_intersect:
        arr = []
        for i in range(np.size(sspSolution, 0) - 1):
            if UPoincare(sspSolution[i],sspTemplate,nTemplate) \
            < 0 < UPoincare(sspSolution[i+1],sspTemplate,nTemplate):
                ax.scatter(sspSolution[i,0],sspSolution[i,1],sspSolution[i,2],
                           'o',color='b',alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlabel('X(t)')
    ax.set_ylabel(r'X(t+$\tau$)')
    ax.set_zlabel(r'X(t+2$\tau$)')
    ax.view_init(30, 35)


def poincare_intersection(sspSolution,sspTemplate,nTemplate):
    '''
    Finds points in trajectory which intersect poincare section
    
    Inputs:
           - sspSolution: Time-delay embedding
           - sspTemplate: Point on hyperplane
           - nTemplate: Plane normal
    Outputs:
            - sspSolution: Array of every intersection with x,y and z coords
    '''    
    sspSolutionPoincare = np.array([], float)
    for i in range(np.size(sspSolution, 0) - 1):
        # Look at every timepoint to test for hyperplane crossings
        if UPoincare(sspSolution[i],sspTemplate,nTemplate) \
            < 0 < UPoincare(sspSolution[i+1],sspTemplate,nTemplate):

            # Uses the halfway point between both points as rough estimate of intersection
            sspPoincare = [((sspSolution[i][0]+sspSolution[i+1][0])/2),
                          ((sspSolution[i][1]+sspSolution[i+1][1])/2),
                          ((sspSolution[i][2]+sspSolution[i+1][2])/2)]

            # Adds coord to new array 
            sspSolutionPoincare = np.append(sspSolutionPoincare, sspPoincare)
    
    #Reshapes above into array of 3d vectors
    sspSolutionPoincare = sspSolutionPoincare.reshape(
                                            int(np.size(sspSolutionPoincare, 0) / 3),
                                                      3)
    
    return sspSolutionPoincare


def project_poincare(sspSolutionPoincare,plot=False):
    '''
    Projects poincare points onto 2d plane where x corresponds 
    to sqrt(X^2+Y^2) and Z corresponds to original Z-value
    
    Inputs: 
            - sspSolutionPoincare: Points generated with poincare_intersection
            - plot: Boolean indicating whether to plot the projection
            
    Outputs:
            - PoincareSection: 2d projection of intersect points
    '''
    
    #Create new arrays witjh projection
    PoincareSection = np.zeros((len(sspSolutionPoincare),2))
    PoincareSection[:,0] = np.sqrt(sspSolutionPoincare[:,0]**2+sspSolutionPoincare[:,1]**2)
    PoincareSection[:,1] = sspSolutionPoincare[:,2]
    
    if plot:
        fig = plt.figure(figsize=(10,6))
        ax = fig.gca()
        plt.plot(PoincareSection[:,0],PoincareSection[:,1],'bo',markersize=5)
        ax.set_xlabel('$\sqrt{X(t)^2 + X(t+𝜏)^2}\'$')
        ax.set_xlim([-0.75,1.3])
        ax.set_ylim([-0.3,0.7])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    return PoincareSection


def project_poincare_v2(sspSolutionPoincare,sspTemplate,nTemplate):
    '''
    Projects state-space vectors onto poincare section
    
    Inputs:
            - sspSolutionPoincare: vector generated with poincare_intersection
            - sspTemplate
            - ntemplate
    Outputs:
            - PoincareSection: Projection of intersection points
    '''
    
    # Constructs a matrix which projects state space vectors onto these basis
    e_z = np.array([0, 0, 1], float)  # Unit vector in z direction
    ProjPoincare = np.array([sspTemplate,
                             e_z,
                             nTemplate], float)
    # sspSolutionPoincare has state space vectors on its rows. We act on the
    # transpose of this matrix to project each state space point onto Poincare
    # basis by a simple matrix multiplication
    PoincareSection = np.dot(ProjPoincare, sspSolutionPoincare.transpose())
    # We return to the usual N x 3 form by another transposition
    return PoincareSection.transpose() 


def pc_sig(sig,fs,start,stop,theta,plot_te=True,plot_pc=True):
    '''
    Poincare Section of Signal
        - Runs above functions on signal to produce final Poincare intersection output
    Inputs:
            - sig: PPG/ECG signal
            - fs: Sampling frequency
            - start: Start index
            - stop: Stop index
            - theta: Angle of section
            - plot_te: Boolean indicating whether to plot time embedding
            - plot_pc: Boolean indicating whether to plot Poincare section
    
    Outputs:
            - PoincareSection: Poincare section    
    '''
    
    #Get hyperplane params
    sspTemplate,nTemplate = get_plane_vec(theta)
    
    #Delay embed signal
    sspSolution = delay_embed(sig,fs,start,stop)
    
    #Plots time embedding if true
    if plot_te:
        plot_embed(sspSolution,True,sspTemplate,nTemplate)
    
    #Get intersections
    sspSolutionPoincare = poincare_intersection(sspSolution,sspTemplate,nTemplate)
    
    #Get Poincare section
    PoincareSection = project_poincare(sspSolutionPoincare,plot_pc)
    
    return PoincareSection
