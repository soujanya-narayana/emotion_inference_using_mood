import os
import logging
import torch
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import yaml
import seaborn as sns


def load_config(config_file_path):
    """
    Function to load configuration file
    """
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    return config


def dump_config(config, path_to_save):
    """
    Function to dump configuration file
    """
    with open(os.path.join(path_to_save, 'config.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping():
    """
    Adopted from https://stackoverflow.com/a/71999355
    """
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's validation loss is less than the previous
    least loss, then save the model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    # def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, save_model_dir, logger, fold_num):
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, save_model_dir, logger):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            logger.info(f'Best validation loss: {round(self.best_valid_loss, 4)}')
            logger.info(f'Saving best model for epoch: {epoch}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion},
                # os.path.join(save_model_dir, f'best_model_{fold_num+1}.pth')) # Use when performing 5-fold cv
                os.path.join(save_model_dir, f'best_model.pth'))


class SaveBestModelCCC:
    """
    Class to save the best model while training. If the current epoch's validation loss is less than the previous
    least loss, then save the model state.
    """
    def __init__(self, best_valid_ccc=-float('inf')):
        self.best_valid_ccc = best_valid_ccc

    # def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, save_model_dir, logger, fold_num):
    def __call__(self, current_valid_ccc, epoch, model, optimizer, criterion, save_model_dir, logger):
        if current_valid_ccc > self.best_valid_ccc:
            self.best_valid_ccc = current_valid_ccc
            logger.info(f'Best CCC: {round(self.best_valid_ccc, 4)}')
            logger.info(f'Saving best model for epoch: {epoch}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion},
                # os.path.join(save_model_dir, f'best_model_{fold_num+1}.pth')) # Use when performing 5-fold cv
                os.path.join(save_model_dir, f'best_model.pth'))


# def save_model(epoch, model, optimizer, criterion, save_model_dir, logger, fold_num):
def save_model(epoch, model, optimizer, criterion, save_model_dir, logger):
    """
    Function to save the trained model to disk
    """
    logger.info(f'Saving the model...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion},
        # os.path.join(save_model_dir, f'final_model_{fold_num+1}.pth')) # Use when performing 5-fold cv
        os.path.join(save_model_dir, f'model_{epoch}.pth'))


def get_logger(save_path):

    print('Creating Log')
    # create a logger object instance
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(logging.INFO)
    file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)
    return logger


def conf_mat(y_true, y_pred, labels, display_labels, savefig_path=None):
    """
    Plot confusion matrix for similar/dissimilar predictions.
    :param y_true:
    :param y_pred:
    :param savefig_path:
    :return:
    """
    plt.close("all")
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap='YlGnBu')
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.show()
    plt.close()
    return cm

'''
def conf_mat(cm_list, savefig_path=None):
    """
    Plot confusion matrix for mood predictions.
    :param cm_list: confusion matrix of all folds
    :param savefig_path:
    :return:
    """
    plt.close("all")
    conf_mat = np.mean(cm_list, axis=0)
    ax = sns.heatmap(conf_mat, cmap='YlGnBu', annot=True, fmt='.2f', vmin=0, vmax=1, annot_kws={'fontsize': 14})
    labels = ['Negative', 'Neutral', 'Positive']
    ax.set_yticklabels(labels, rotation=0, fontsize=12)
    ax.set_xticklabels(labels, rotation=0, fontsize=12)
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.show()
'''

# For visualising plots
# Works like tensorboard
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name, save_dir=None):
        self.viz = Visdom()
        self.env = env_name
        self.save_dir = save_dir
        self.plots = {}


    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def save_json(self):
        #
        self.viz.save([self.env])

        if self.save_dir is not None:
            import shutil
            p = shutil.copy2(os.path.join(os.path.expanduser("~/.visdom"), self.env + ".json"), self.save_dir)

    def conf_mat(self, y_true, y_pred, labels, display_labels, epoch, title):
        assert self.save_dir is not None

        if not os.path.isdir(os.path.join(self.save_dir, "conf_mats")):
            os.makedirs(os.path.join(self.save_dir, "conf_mats"), exist_ok=True)

        plt.close("all")
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot()

        save_img_path = os.path.join(self.save_dir, "conf_mats", f"conf_mat_epoch{epoch}.jpeg")
        plt.savefig(save_img_path, bbox_inches='tight')
        img = plt.imread(save_img_path)
        img = np.moveaxis(img, -1, 0)
        self.viz.image(img,
                       win='Confusion Matrix',
                       opts=dict(caption=f'Epoch {epoch}', store_history=True,
                                 title=title),
                       env=self.env
                       )