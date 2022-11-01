from math import log10, sqrt
import cv2
import numpy as np
import pandas as pd 

import os 
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr 
from skimage.metrics import mean_squared_error
from skimage import data, img_as_float

def PSNR(original, compressed):
    max_pixel = 1
    maxv=np.max(original)
    minv=np.min(original)
    
    original = (original - minv) / (maxv - minv)
    compressed = (compressed - minv) / (maxv - minv)
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def SSI_MSE (original, predicted):
    original = img_as_float(original)
    predicted = img_as_float(predicted)
    mse_ = mean_squared_error(original, predicted)
    ssim_ = ssim(np.squeeze(original), np.squeeze(predicted), data_range=None, multichannel= True)

    return mse_,ssim_

def evaluate_model(model,test_dataset, experiment_dir):
    psnr_pr = []
    ssi_pr=[]
    
    mse_pr=[]
    for inp, tar in test_dataset:
        predimg = model(inp, training=True)  
        trueimg=tar
        
        psnr_pr_vs_real = PSNR(trueimg, predimg)

        psnr_pr.append(psnr_pr_vs_real)
        mse_pr_vs_real,ssi_pr_vs_real = SSI_MSE(trueimg, predimg)
        ssi_pr.append(ssi_pr_vs_real),mse_pr.append(mse_pr_vs_real)
        
        outpath=experiment_dir
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    
    evaluation_report=[psnr_pr,ssi_pr,mse_pr]
    evaluation_report=pd.DataFrame(np.array(evaluation_report).T,columns=['SNRpr','SSIpr','MSEpr'])
        
    return evaluation_report