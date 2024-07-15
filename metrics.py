import numpy as np
import torch


def ACC(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    return np.mean(ground_truth.astype(int) == predictions.astype(int))


def RMSE(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    return np.sqrt(np.mean((ground_truth-predictions)**2))


def SAGR(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    return np.mean(np.sign(ground_truth) == np.sign(predictions))


def PCC(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    return np.nan_to_num(np.corrcoef(ground_truth, predictions)[0,1])


def CCC(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)


def ccc(x, y):
    pcc = torch.corrcoef(torch.stack((x, y), dim=0))[0, 1]
    num = 2 * pcc * x.std() * y.std()
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    ccc = num / den
    return torch.nan_to_num(ccc, nan=0)


def dyn_wt_mse_ccc_loss(x, y, epoch, max_epochs, weight_exponent=2, clamp=False):
    # give more weights to CCC at the initial training phase, then to mse

    weights = ((epoch/max_epochs)**weight_exponent, 1.0 - ((epoch/max_epochs)**weight_exponent))

    if clamp:
        x, y = x.clamp(-1, 1), y.clamp(-1, 1)

    loss = (weights[0] * torch.nn.functional.mse_loss(x, y)) +\
           (weights[1] * (1.0 - ccc(x, y)))

    return loss


def ccc_loss(x, y, clamp=False):
    # x and y shape: (bs, 2)
    # first dimension for valence, second for arousal

    if clamp:
        val_pred, val_true = x[:, 0].clamp(-1, 1), y[:, 0].clamp(-1, 1)
        
    ccc_v = ccc(x, y)

    loss = 1 - ccc_v

    return loss
