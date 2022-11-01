import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def plotVox(data, mode = 0, clim = (None, None), xcut = -1, ycut = -1, zcut = -1):
    #mode = 1 for simset, mode = 0 for default
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    fig.tight_layout()
    if xcut == -1:
        xcut = len(data)//2
    if ycut == -1:
        ycut = len(data[0])//2
    if zcut == -1:
        zcut = len(data[0][0])//2
    for p in range(3): # for different planes
        im = None
        if p == 0:
            #im = ax[p].imshow(np.flip(np.transpose(data[:,:,zcut], (1,0)), axis=0), cmap=plt.cm.gray)
            im = ax[p].imshow(data[:,:,zcut], cmap=plt.cm.gray)
            ax[p].title.set_text("XY plane")
        if p == 1:
            im = ax[p].imshow(np.flip(np.transpose(data[:,ycut,:], (1,0)), axis=0), cmap=plt.cm.gray)
            ax[p].title.set_text("XZ plane")
        if p == 2:
            im = ax[p].imshow(np.flip(np.transpose(data[xcut,:,:], (1,0)), axis=0), cmap=plt.cm.gray)
            ax[p].title.set_text("YZ plane")
        if mode == 1: # for simset
            if p == 0:
                im = ax[p].imshow(np.transpose(data[:,:,zcut], (1,0)), cmap=plt.cm.gray)
                ax[p].title.set_text("XY plane")
            if p == 1:
                im = ax[p].imshow(np.flip(np.transpose(data[xcut,:,:], (1,0)), axis=0), cmap=plt.cm.gray)
                ax[p].title.set_text("YZ plane")
            if p == 2:
                im = ax[p].imshow(np.flip(np.transpose(data[:,ycut,:], (1,0)), axis=0), cmap=plt.cm.gray)
                ax[p].title.set_text("XZ plane")
        fig.colorbar(im, ax=ax[p])
        if clim != (None, None):
            im.set_clim(clim[0],clim[1])
    plt.show()

def plotSimSET(filename, voxNumTuple, clim = (None, None), xcut = -1, ycut = -1, zcut = -1):
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype = np.int32)
        data = np.reshape(data, voxNumTuple, order='F')
        plotVox(data, mode=1, clim=clim, xcut=xcut, ycut=ycut, zcut=zcut)

def readSimSET(filename, voxNumTuple):
    data = None
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype = np.int32)
    return np.reshape(data, voxNumTuple, order='F')


def getNRMSE(A, B):
    # Assume A and B are numpy array
    # RMSE will be normalized by mean of B
    assert(A.size == B.size)
    m = A.size
    return np.sqrt(np.sum((A - B)**2)/m)/(np.mean(B))

def getPSNR(A, B):
    # Assume A is input, B is target
    mse = mean_squared_error(A, B)
    return 20*np.log10(np.max(B)) - 10*np.log10(mse)
