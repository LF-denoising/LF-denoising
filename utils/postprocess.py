import numpy as np

def norm(x):
    # normalise x to range [-1,1]
    nom = (x - x.min()) * 2.0
    denom = x.max() - x.min()
    return  nom/denom - 1.0

def standardizate(x):
    return (x - x.mean()) / x.std()

def sigmoid(x, k=0.85):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + np.exp(-x / k)) 
    return s

def postprocess(sub_images, fusion_im, q_vmin=0.05, q_vmax=1, sig=0.4, fix_t=False, use_avg=False):
    
    avg_sub = sum(sub_images) / len(sub_images)
    
    if use_avg:
        return avg_sub * 0.9 + fusion_im * 0.1
    
    thres = np.quantile(avg_sub[avg_sub != 0], q_vmin)
    thres_max = np.quantile(avg_sub[avg_sub != 0], q_vmax)
    avg_sub = np.clip(avg_sub, thres, thres_max)
    if sig is not None:
        avg_sub = norm(avg_sub)
        avg_sub = sigmoid(avg_sub, k=sig)
        if fix_t:
            avg_sub *= (thres_max - thres)
        final = fusion_im * avg_sub
        final *= fusion_im.mean() / final.mean()
    else:
        final = fusion_im * (avg_sub - avg_sub.min()) / (avg_sub.max() - avg_sub.min())
    return final