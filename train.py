from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold, StratifiedKFold,\
    train_test_split, KFold
from sklearn.metrics import confusion_matrix, f1_score
from collections import Counter
from dataloader import MoodEmo, MoodEmotion
import torch.optim as optim
from model import get_model
from metrics import RMSE, PCC, CCC, SAGR, ACC, dyn_wt_mse_ccc_loss, ccc_loss
from utils import AverageMeter, VisdomLinePlotter, save_model, SaveBestModel, conf_mat, SaveBestModelCCC

torch.manual_seed(5)


def train_mood_delta_emonet(model, data_root, train_df, val_df, test_df, cfg, logger, device, plotter, save_dir, test_model):
    train_dataset = MoodEmo(data_root, train_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, sampler=None,
                              num_workers=cfg["NUM_WORKERS"], drop_last=True)

    val_dataset = MoodEmo(data_root, val_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg["GAMMA"])
    tr_loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg["MOOD_WTS"]).to(device))  # [1.44, 4.34, 12.5]
    tr_loss_func2 = torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg["DELTA_WTS"]).to(device))  # [11.11, 2.22, 2.22]
    tr_loss_func3 = torch.nn.MSELoss()
    val_loss_func1 = torch.nn.CrossEntropyLoss()
    val_loss_func2 = torch.nn.CrossEntropyLoss()
    val_loss_func3 = torch.nn.MSELoss()
    save_best_model = SaveBestModel()
    train_loss_mood = AverageMeter()
    train_loss_delta = AverageMeter()
    train_loss_valence = AverageMeter()
    total_train_loss = AverageMeter()
    val_loss_mood = AverageMeter()
    val_loss_delta = AverageMeter()
    val_loss_valence = AverageMeter()
    total_val_loss = AverageMeter()
    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence', 'Accuracy_mood', 'Accuracy_delta',
            'F1Score_mood', 'F1Score_delta']
    train_metrics = {key: AverageMeter() for key in keys}
    wt = (0.2, 0.2, 1.0)
    for epoch in range(1, cfg["NUM_EPOCHS"]+1):
        logger.info(f'Epoch {epoch} / {cfg["NUM_EPOCHS"]}')
        # Train phase
        running_train_loss = 0.0
        model.train()
        for tr_batch_idx, train_dict in enumerate(train_loader):
            x_train_mood = train_dict['frames'].to(device)
            # x_train_mood_previous = train_dict['video_previous_mood_vec'].to(device)
            # print('Clip:', x_train_mood.shape)
            y_train_mood = train_dict['mood'].to(device)
            y_train_delta = train_dict['delta_val'].to(device)
            # print("y train mood:", y_train_mood)
            x_train_img = train_dict['current_image'].to(device)
            # print('Image shape', x_train_img.shape)
            y_train_img = train_dict['current_valence'].to(device)
            # print('y_train shape:', y_train_img.shape)
            # logger.info(f'tr_batch_idx: {tr_batch_idx}')
            optimizer.zero_grad()
            out_dict = model(x_train_mood, x_train_img)
            # print(tr_pred_img.shape)
            tr_pred_mood = out_dict['mood'].to(device)
            tr_pred_delta = out_dict['delta'].to(device)
            tr_pred_img = torch.tanh(out_dict['valence'].flatten()).to(device)
            # print('tr_pred shape:', tr_pred.shape)
            loss1 = tr_loss_func1(tr_pred_mood, y_train_mood)
            loss2 = tr_loss_func2(tr_pred_delta, y_train_delta)
            # print('loss1:', loss1)
            loss3 = dyn_wt_mse_ccc_loss(tr_pred_img.float(), y_train_img.float(), epoch=epoch, max_epochs=cfg['NUM_EPOCHS'], clamp=False)
            # print('loss1:', loss1)
            # loss3 = tr_loss_func3(tr_pred_img.float(), y_train_img.float())
            loss = wt[0] * loss1 + wt[1] * loss2 + wt[2] * loss3
            # print('Loss:', loss)
            loss.backward()
            optimizer.step()

            # Log
            running_train_loss += loss.item()
            train_loss_mood.update(loss1.item(), y_train_mood.size(0))
            train_loss_delta.update(loss2.item(), y_train_delta.size(0))
            train_loss_valence.update(loss3.item(), y_train_img.size(0))
            total_train_loss.update(loss.item(), y_train_mood.size(0))
            RMSE_valence = RMSE(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(),
                                y_train_img.clone().detach().cpu().numpy())
            PCC_valence = PCC(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(),
                              y_train_img.clone().detach().cpu().numpy())
            CCC_valence = CCC(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(),
                              y_train_img.clone().detach().cpu().numpy())
            SAGR_valence = SAGR(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(),
                                y_train_img.clone().detach().cpu().numpy())
            acc_mood = ACC(torch.max(out_dict['mood'], dim=1)[1].clone().detach().cpu().numpy(),
                           y_train_mood.clone().detach().cpu().numpy())
            # print(acc_mood)
            acc_delta = ACC(torch.max(out_dict['delta'], dim=1)[1].clone().detach().cpu().numpy(),
                            y_train_delta.clone().detach().cpu().numpy())
            F1Score_mood = f1_score(torch.max(out_dict['mood'], dim=1)[1].clone().detach().cpu().numpy(),
                                    y_train_mood.clone().detach().cpu().numpy(), average='weighted')
            F1Score_delta = f1_score(torch.max(out_dict['delta'], dim=1)[1].clone().detach().cpu().numpy(),
                                     y_train_delta.clone().detach().cpu().numpy(), average='weighted')

            train_metrics['RMSE_valence'].update(RMSE_valence, y_train_img.size(0))
            train_metrics['PCC_valence'].update(PCC_valence, y_train_img.size(0))
            train_metrics['CCC_valence'].update(CCC_valence, y_train_img.size(0))
            train_metrics['SAGR_valence'].update(SAGR_valence, y_train_img.size(0))
            train_metrics['Accuracy_mood'].update(acc_mood, y_train_mood.size(0))
            train_metrics['Accuracy_delta'].update(acc_delta, y_train_delta.size(0))
            train_metrics['F1Score_mood'].update(F1Score_mood, y_train_mood.size(0))
            train_metrics['F1Score_delta'].update(F1Score_delta, y_train_delta.size(0))

            if (tr_batch_idx + 1) % cfg["LOG_INTERVAL"] == 0:
                avg_train_loss = running_train_loss / cfg["LOG_INTERVAL"]
                logger.info('Epoch {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}\t'
                            .format(epoch, (tr_batch_idx + 1) * len(x_train_mood), len(train_loader.dataset),
                                    100 * tr_batch_idx / len(train_loader), avg_train_loss))
                running_train_loss = 0.0
        logger.info('Train set ({:d} samples): Mood loss: {:.4f}\tDelta loss: {:.4f}\tvalence loss: {:.4f}'.format(
            len(train_loader.dataset), train_loss_mood.avg, train_loss_delta.avg, train_loss_valence.avg))
        logger.info('Total train loss: {:.4f}'.format(total_train_loss.avg))
        # logger.info('Total train loss: {:.4f}'.format(len(train_loader.dataset), total_train_loss.avg))
        logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            train_metrics['RMSE_valence'].avg, train_metrics['PCC_valence'].avg, train_metrics['CCC_valence'].avg,
            train_metrics['SAGR_valence'].avg))
        logger.info('Acc Mood: {:.4f}\tF1Score Mood: {:.4f}'.format(
            train_metrics['Accuracy_mood'].avg, train_metrics['F1Score_mood'].avg))
        logger.info('Acc Delta: {:.4f}\tF1Score Delta: {:.4f}'.format(
            train_metrics['Accuracy_delta'].avg, train_metrics['F1Score_delta'].avg))

        plotter.plot('Loss (Mood)', 'train', 'Mood Loss', epoch, train_loss_mood.avg)
        plotter.plot('Loss (Delta)', 'train', 'Delta Loss', epoch, train_loss_delta.avg)
        plotter.plot('Loss (Reg)', 'train', 'Valence Loss', epoch, train_loss_valence.avg)
        plotter.plot('Loss (Total)', 'train', 'Total Loss', epoch, total_train_loss.avg)
        plotter.plot('RMSE (Valence)', 'train', 'RMSE', epoch, train_metrics['RMSE_valence'].avg)
        plotter.plot('PCC (Valence)', 'train', 'PCC', epoch, train_metrics['PCC_valence'].avg)
        plotter.plot('CCC (Valence)', 'train', 'CCC', epoch, train_metrics['CCC_valence'].avg)
        plotter.plot('SAGR (Valence)', 'train', 'SAGR', epoch, train_metrics['SAGR_valence'].avg)
        plotter.plot('Accuracy', 'train', 'Acc Mood', epoch, train_metrics['Accuracy_mood'].avg)
        plotter.plot('Accuracy', 'train', 'Acc Delta', epoch, train_metrics['Accuracy_delta'].avg)

        # Validation phase
        model.eval()
        val_true_mood = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_mood = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_true_delta = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_delta = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_true_valence = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_valence = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence', 'Accuracy_mood', 'Accuracy_delta',
                'F1Score_mood', 'F1Score_delta']
        val_metrics = {key: 0 for key in keys}
        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(tqdm(val_loader)):
                x_val_mood = val_dict['frames'].to(device)
                # x_val_mood_previous = val_dict['video_previous_mood_vec'].to(device)
                # x_val_delta = val_dict['video_delta_vector'].to(device)
                y_val_mood = val_dict['mood'].to(device)
                y_val_delta = val_dict['delta_val'].to(device)
                x_val_img = val_dict['current_image'].to(device)
                y_val_img = torch.tanh(val_dict['current_valence'].flatten()).to(device)
                # print('y_val shape:', y_val.shape)

                out_dict = model(x_val_mood, x_val_img)
                # print('val_pred shape:', out_dict['out'].ravel().to(device).shape)
                loss1 = val_loss_func1(out_dict['mood'], y_val_mood)
                loss2 = val_loss_func2(out_dict['delta'], y_val_delta)
                val_pred_img = torch.tanh(out_dict['valence'].flatten()).to(device)
                # loss3 = val_loss_func3(out_dict['valence'].ravel(), y_val_img)
                loss3 = dyn_wt_mse_ccc_loss(val_pred_img.float(), y_val_img.float(), epoch=epoch,
                                             max_epochs=cfg['NUM_EPOCHS'], clamp=False)
                loss = wt[0] * loss1 + wt[1] * loss2 + wt[2] * loss3
                val_loss_mood.update(loss1.item(), y_val_mood.size(0))
                val_loss_delta.update(loss2.item(), y_val_delta.size(0))
                val_loss_valence.update(loss3.item(), y_val_img.size(0))
                total_val_loss.update(loss.item(), y_val_mood.size(0))

                val_true_mood[val_batch_idx, :] = y_val_mood.clone().detach().cpu().numpy()
                val_pred_mood[val_batch_idx, :] = torch.max(out_dict['mood'], dim=1)[1].clone().detach().cpu().numpy()

                val_true_delta[val_batch_idx, :] = y_val_delta.clone().detach().cpu().numpy()
                val_pred_delta[val_batch_idx, :] = torch.max(out_dict['delta'], dim=1)[1].clone().detach().cpu().numpy()

                val_true_valence[val_batch_idx, :] = y_val_img.clone().detach().cpu().numpy()
                val_pred_valence[val_batch_idx, :] = torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy()

            val_true_mood = np.squeeze(np.asarray(val_true_mood)).flatten()
            val_pred_mood = np.squeeze(np.asarray(val_pred_mood)).flatten()

            val_true_delta = np.squeeze(np.asarray(val_true_delta)).flatten()
            val_pred_delta = np.squeeze(np.asarray(val_pred_delta)).flatten()

            val_true_valence = np.squeeze(np.asarray(val_true_valence)).flatten()
            val_pred_valence = np.squeeze(np.asarray(val_pred_valence)).flatten()

        val_metrics['RMSE_valence'] = RMSE(val_true_valence, val_pred_valence)
        val_metrics['PCC_valence'] = PCC(val_true_valence, val_pred_valence)
        val_metrics['CCC_valence'] = CCC(val_true_valence, val_pred_valence)
        val_metrics['SAGR_valence'] = SAGR(val_true_valence, val_pred_valence)
        val_metrics['Accuracy_mood'] = ACC(val_true_mood, val_pred_mood)
        val_metrics['Accuracy_delta'] = ACC(val_true_delta, val_pred_delta)
        val_metrics['F1Score_mood'] = f1_score(val_true_mood, val_pred_mood, average='weighted')
        val_metrics['F1Score_delta'] = f1_score(val_true_delta, val_pred_delta, average='weighted')

        logger.info('Val set ({:d} samples): Mood loss: {:.4f}\tDelta loss: {:.4f}\tValence loss: {:.4f}'.format(
            len(val_loader.dataset), val_loss_mood.avg, val_loss_delta.avg, val_loss_valence.avg))
        # logger.info('Total val loss: {:.4f}'.format(len(val_loader.dataset), total_val_loss.avg))
        logger.info('Total val loss: {:.4f}'.format(total_val_loss.avg))
        logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            val_metrics['RMSE_valence'], val_metrics['PCC_valence'], val_metrics['CCC_valence'],
            val_metrics['SAGR_valence']))
        logger.info('Acc Mood: {:.4f}\tF1Score Mood: {:.4f}'.format(val_metrics['Accuracy_mood'], val_metrics['F1Score_mood']))
        logger.info('Acc Delta: {:.4f}\tF1Score Delta: {:.4f}'.format(
            val_metrics['Accuracy_delta'], val_metrics['F1Score_delta']))

        plotter.plot('Loss (Mood)', 'val', 'Mood Loss', epoch, val_loss_mood.avg)
        plotter.plot('Loss (Delta)', 'val', 'Delta Loss', epoch, val_loss_delta.avg)
        plotter.plot('Loss (Reg)', 'val', 'Valence Loss', epoch, val_loss_valence.avg)
        plotter.plot('Loss (Total)', 'val', 'Total Loss', epoch, total_val_loss.avg)
        plotter.plot('RMSE (Valence)', 'val', 'RMSE', epoch, val_metrics['RMSE_valence'])
        plotter.plot('PCC (Valence)', 'val', 'PCC', epoch, val_metrics['PCC_valence'])
        plotter.plot('CCC (Valence)', 'val', 'CCC', epoch, val_metrics['CCC_valence'])
        plotter.plot('SAGR (Valence)', 'val', 'SAGR', epoch, val_metrics['SAGR_valence'])
        plotter.plot('Accuracy', 'val', 'Acc Mood', epoch, val_metrics['Accuracy_mood'])
        plotter.plot('Accuracy', 'val', 'Acc Delta', epoch, val_metrics['Accuracy_delta'])
        # save_model(epoch, model, optimizer, val_loss_func2, save_dir, logger)

        # Save the best model of this fold
        save_best_model(total_val_loss.avg, epoch, model, optimizer, val_loss_func2, os.path.join(save_dir), logger)
        scheduler.step()

        # Save the final model of this fold
    save_model(epoch, model, optimizer, val_loss_func2, save_dir, logger)
    del model

    # Test using the best model
    test_dataset = MoodEmo(data_root, test_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TEST_BATCH_SIZE"], shuffle=False,
                             num_workers=cfg["NUM_WORKERS"], drop_last=True)

    checkpoint = torch.load(os.path.join(save_dir, f'best_model.pth'))
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()

    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence', 'Accuracy_mood', 'Accuracy_delta',
            'F1Score_mood', 'F1Score_delta']
    test_metrics = {key: 0 for key in keys}
    target_true_mood = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_mood = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_true_delta = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_delta = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_true_valence = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_valence = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))

    with torch.no_grad():
        for test_batch_idx, test_dict in enumerate(tqdm(test_loader)):
            x_test_mood = test_dict['frames'].to(device)
            # x_test_mood_previous = test_dict['video_previous_mood_vec'].to(device)
            # x_test_delta = test_dict['video_delta_vec'].to(device)
            y_test_mood = test_dict['mood'].to(device)
            y_test_delta = test_dict['delta_val'].to(device)
            x_test_img = test_dict['current_image'].to(device)
            y_test_img = test_dict['current_valence'].flatten().to(device)
            # print('y_test shape:', y_test.shape)

            out_dict = test_model(x_test_mood, x_test_img)
            # print('test_pred shape:', out_dict['out'].ravel().to(device).shape)

            target_true_mood[test_batch_idx, :] = y_test_mood.clone().detach().cpu().numpy()
            target_pred_mood[test_batch_idx, :] = torch.max(out_dict['mood'], dim=1)[1].clone().detach().cpu().numpy()
            target_true_delta[test_batch_idx, :] = y_test_delta.clone().detach().cpu().numpy()
            target_pred_delta[test_batch_idx, :] = torch.max(out_dict['delta'], dim=1)[1].clone().detach().cpu().numpy()
            target_true_valence[test_batch_idx, :] = y_test_img.clone().detach().cpu().numpy()
            target_pred_valence[test_batch_idx, :] = torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy()
            # print(test_pred)

    target_true_mood = np.squeeze(np.asarray(target_true_mood)).flatten()
    target_pred_mood = np.squeeze(np.asarray(target_pred_mood)).flatten()
    target_true_delta = np.squeeze(np.asarray(target_true_delta)).flatten()
    target_pred_delta = np.squeeze(np.asarray(target_pred_delta)).flatten()
    target_true_valence = np.squeeze(np.asarray(target_true_valence)).flatten()
    target_pred_valence = np.squeeze(np.asarray(target_pred_valence)).flatten()

    test_metrics['RMSE_valence'] = RMSE(target_true_valence, target_pred_valence)
    test_metrics['PCC_valence'] = PCC(target_true_valence, target_pred_valence)
    test_metrics['CCC_valence'] = CCC(target_true_valence, target_pred_valence)
    test_metrics['SAGR_valence'] = SAGR(target_true_valence, target_pred_valence)
    test_metrics['Accuracy_mood'] = ACC(target_true_mood, target_pred_mood)
    test_metrics['Accuracy_delta'] = ACC(target_true_delta, target_pred_delta)
    test_metrics['F1Score_mood'] = f1_score(target_true_mood, target_pred_mood, average='weighted')
    test_metrics['F1Score_delta'] = f1_score(target_true_delta, target_pred_delta, average='weighted')

    logger.info('Test set ({:d} samples): \t'.format(len(test_loader.dataset)))
    logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
        test_metrics['RMSE_valence'], test_metrics['PCC_valence'], test_metrics['CCC_valence'],
        test_metrics['SAGR_valence']))
    logger.info('Acc Mood: {:.4f}\tF1Score Mood: {:.4f}'.format(test_metrics['Accuracy_mood'], test_metrics['F1Score_mood']))
    logger.info('Acc Delta: {:.4f}\tF1Score Delta: {:.4f}'.format(test_metrics['Accuracy_delta'], test_metrics['F1Score_delta']))

    cm_mood = conf_mat(y_true=target_true_mood, y_pred=target_pred_mood, labels=[2, 0, 1],
                       display_labels=["Negative", "Neutral", "Positive"],
                       savefig_path=os.path.join(save_dir, f'conf_mat_mood.png'))

    cm_delta = conf_mat(y_true=target_true_delta, y_pred=target_pred_delta, labels=[2, 0, 1],
                        display_labels=["Negative", "Neutral", "Positive"],
                        savefig_path=os.path.join(save_dir, f'conf_mat_delta.png'))

    return dict(train_metrics=train_metrics, train_loss=total_train_loss.avg, val_loss=total_val_loss.avg,
                y_true_mood=target_true_mood, y_pred_mood=target_pred_mood, y_true_valence=target_true_valence,
                y_pred_valence=target_pred_valence, val_metrics=val_metrics, test_metrics=test_metrics,
                conf_mat_mood=cm_mood, conf_mat_delta=cm_delta)


def train_resnet_emonet(model, data_root, train_df, val_df, test_df, cfg, logger, device, plotter, save_dir, test_model):
    train_dataset = MoodEmo(data_root, train_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, sampler=None,
                              num_workers=cfg["NUM_WORKERS"], drop_last=True)

    val_dataset = MoodEmo(data_root, val_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg["GAMMA"])
    # mood_wts = torch.tensor([1.44, 4.34, 12.5]).to(device)
    # delta_wts = torch.tensor([11.11, 2.22, 2.22]).to(device)
    tr_loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg["MOOD_WTS"]).to(device))
    val_loss_func1 = torch.nn.CrossEntropyLoss()
    loss_func2 = torch.nn.MSELoss()
    save_best_model = SaveBestModel()
    train_loss_valence = AverageMeter()
    train_loss_mood = AverageMeter()
    total_train_loss = AverageMeter()
    val_loss_mood = AverageMeter()
    val_loss_valence = AverageMeter()
    total_val_loss = AverageMeter()
    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence', 'Accuracy', 'F1Score']
    train_metrics = {key: AverageMeter() for key in keys}
    wt = (0.2, 1.0)
    for epoch in range(1, cfg["NUM_EPOCHS"]+1):
        logger.info(f'Epoch {epoch} / {cfg["NUM_EPOCHS"]}')
        # Train phase
        running_train_loss = 0.0
        model.train()
        for tr_batch_idx, train_dict in enumerate(train_loader):
            x_train_mood = train_dict['frames'].to(device)
            # x_train_mood_previous = train_dict['video_previous_mood_vec'].to(device)
            # print('Clip:', x_train_mood.shape)
            y_train_mood = train_dict['mood'].to(device)  # use 'delta_val' to predict delta delta, 'mood' to predict mood
            # print("y train mood:", y_train_mood)
            x_train_img = train_dict['current_image'].to(device)
            # print('Image shape', x_train_img.shape)
            y_train_img = train_dict['current_valence'].to(device)
            # print('y_train shape:', y_train_img.shape)
            # logger.info(f'tr_batch_idx: {tr_batch_idx}')
            optimizer.zero_grad()
            out_dict = model(x_train_mood, x_train_img)
            tr_pred_img = out_dict['valence'].flatten().to(device)
            # print(tr_pred_img.shape)
            tr_pred_mood = out_dict['pred'].to(device)
            # print(tr_pred_mood.shape)
            # print('tr_pred shape:', tr_pred.shape)
            loss1 = tr_loss_func1(tr_pred_mood, y_train_mood)
            # print('loss1:', loss1)
            # loss2 = loss_func2(tr_pred_img.float(), y_train_img.float())
            loss2 = dyn_wt_mse_ccc_loss(tr_pred_img.float(), y_train_img.float(), epoch=epoch, max_epochs=cfg['NUM_EPOCHS'],
                                        clamp=False)
            # print('loss1:', loss1)
            loss = wt[0] * loss1 + wt[1] * loss2
            # print('Loss:', loss)
            loss.backward()
            optimizer.step()

            # Log
            running_train_loss += loss.item()
            train_loss_mood.update(loss1.item(), y_train_mood.size(0))
            train_loss_valence.update(loss2.item(), y_train_img.size(0))
            total_train_loss.update(loss.item(), y_train_mood.size(0))
            RMSE_valence = RMSE(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(), y_train_img.clone().detach().cpu().numpy())
            PCC_valence = PCC(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(), y_train_img.clone().detach().cpu().numpy())
            CCC_valence = CCC(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(), y_train_img.clone().detach().cpu().numpy())
            SAGR_valence = SAGR(torch.tanh(out_dict['valence']).clone().detach().cpu().numpy(), y_train_img.clone().detach().cpu().numpy())
            acc = ACC(torch.max(out_dict['pred'], dim=1)[1].clone().detach().cpu().numpy(), y_train_mood.clone().detach().cpu().numpy())
            F1Score = f1_score(torch.max(out_dict['pred'], dim=1)[1].clone().detach().cpu().numpy(),
                               y_train_mood.clone().detach().cpu().numpy(), average='weighted')

            train_metrics['RMSE_valence'].update(RMSE_valence, y_train_img.size(0))
            train_metrics['PCC_valence'].update(PCC_valence, y_train_img.size(0))
            train_metrics['CCC_valence'].update(CCC_valence, y_train_img.size(0))
            train_metrics['SAGR_valence'].update(SAGR_valence, y_train_img.size(0))
            train_metrics['Accuracy'].update(acc, y_train_mood.size(0))
            train_metrics['F1Score'].update(F1Score, y_train_mood.size(0))

            if (tr_batch_idx + 1) % cfg["LOG_INTERVAL"] == 0:
                avg_train_loss = running_train_loss / cfg["LOG_INTERVAL"]
                logger.info('Epoch {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t'
                            .format(epoch, (tr_batch_idx + 1) * len(x_train_mood), len(train_loader.dataset),
                                    100 * tr_batch_idx / len(train_loader), avg_train_loss))
                running_train_loss = 0.0
        logger.info('Train set ({:d} samples): Average delta loss: {:.4f}\t'.format(
            len(train_loader.dataset), train_loss_mood.avg))
        logger.info('Train set ({:d} samples): Average valence loss: {:.4f}\t'.format(
            len(train_loader.dataset), train_loss_valence.avg))
        logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}\tAcc: {:.4f}\tF1Score: {:.4f}'.format(
            train_metrics['RMSE_valence'].avg, train_metrics['PCC_valence'].avg, train_metrics['CCC_valence'].avg,
            train_metrics['SAGR_valence'].avg, train_metrics['Accuracy'].avg, train_metrics['F1Score'].avg))

        plotter.plot(f'Loss', 'train', 'Mood Loss', epoch, train_loss_mood.avg)
        plotter.plot(f'Loss', 'train', 'Valence Loss', epoch, train_loss_valence.avg)
        plotter.plot(f'Loss', 'train', 'Total Loss', epoch, total_train_loss.avg)
        plotter.plot('RMSE (Valence)', 'train', 'RMSE', epoch, train_metrics['RMSE_valence'].avg)
        plotter.plot('PCC (Valence)', 'train', 'PCC', epoch, train_metrics['PCC_valence'].avg)
        plotter.plot('CCC (Valence)', 'train', 'CCC', epoch, train_metrics['CCC_valence'].avg)
        plotter.plot('SAGR (Valence)', 'train', 'SAGR', epoch, train_metrics['SAGR_valence'].avg)
        plotter.plot('Accuracy', 'train', 'Acc', epoch, train_metrics['Accuracy'].avg)

        # Validation phase
        model.eval()
        val_true_mood = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_mood = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_true_valence = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_valence = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence', 'Accuracy', 'F1Score']
        val_metrics = {key: 0 for key in keys}
        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(tqdm(val_loader)):
                x_val_mood = val_dict['frames'].to(device)
                # x_val_mood_previous = val_dict['video_previous_mood_vec'].to(device)
                # x_val_delta = val_dict['video_delta_vector'].to(device)
                y_val_mood = val_dict['mood'].to(device)
                x_val_img = val_dict['current_image'].to(device)
                y_val_img = val_dict['current_valence'].flatten().to(device)
                # print('y_val shape:', y_val.shape)

                out_dict = model(x_val_mood, x_val_img)
                # print('val_pred shape:', out_dict['out'].ravel().to(device).shape)
                loss1 = val_loss_func1(out_dict['pred'], y_val_mood)
                # loss2 = loss_func2(out_dict['valence'].ravel(), y_val_img)
                val_pred_img = out_dict['valence'].ravel().to(device)
                loss2 = dyn_wt_mse_ccc_loss(val_pred_img.float(), y_val_img.float(), epoch=epoch, max_epochs=cfg['NUM_EPOCHS'],
                                            clamp=False)
                loss = wt[0] * loss1 + wt[1] * loss2
                val_loss_mood.update(loss1.item(), y_val_mood.size(0))
                val_loss_valence.update(loss2.item(), y_val_img.size(0))
                total_val_loss.update(loss.item(), y_val_mood.size(0))

                val_true_mood[val_batch_idx, :] = y_val_mood.clone().detach().cpu().numpy()
                val_pred_mood[val_batch_idx, :] = torch.max(out_dict['pred'], dim=1)[1].clone().detach().cpu().numpy()

                val_true_valence[val_batch_idx, :] = y_val_img.clone().detach().cpu().numpy()
                val_pred_valence[val_batch_idx, :] = torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy()

            val_true_mood = np.squeeze(np.asarray(val_true_mood)).flatten()
            val_pred_mood = np.squeeze(np.asarray(val_pred_mood)).flatten()

            val_true_valence = np.squeeze(np.asarray(val_true_valence)).flatten()
            val_pred_valence = np.squeeze(np.asarray(val_pred_valence)).flatten()

        val_metrics['RMSE_valence'] = RMSE(val_true_valence, val_pred_valence)
        val_metrics['PCC_valence'] = PCC(val_true_valence, val_pred_valence)
        val_metrics['CCC_valence'] = CCC(val_true_valence, val_pred_valence)
        val_metrics['SAGR_valence'] = SAGR(val_true_valence, val_pred_valence)
        val_metrics['Accuracy'] = ACC(val_true_mood, val_pred_mood)
        val_metrics['F1Score'] = f1_score(val_true_mood, val_pred_mood, average='weighted')

        logger.info('Val set ({:d} samples): Average delta loss: {:.4f}\t'.format(len(val_loader.dataset),
                                                                                val_loss_mood.avg))
        logger.info('Val set ({:d} samples): Average valence loss: {:.4f}\t'.format(len(val_loader.dataset),
                                                                                val_loss_valence.avg))
        logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}\tAcc: {:.4f}\tF1Score: {:.4f}'.format(
            val_metrics['RMSE_valence'], val_metrics['PCC_valence'], val_metrics['CCC_valence'],
            val_metrics['SAGR_valence'], val_metrics['Accuracy'], val_metrics['F1Score']))

        plotter.plot(f'Loss', 'val', 'Mood Loss', epoch, val_loss_mood.avg)
        plotter.plot(f'Loss', 'val', 'Valence Loss', epoch, val_loss_valence.avg)
        plotter.plot(f'Loss', 'val', 'Total Loss', epoch, total_val_loss.avg)
        plotter.plot('RMSE (Valence)', 'val', 'RMSE', epoch, val_metrics['RMSE_valence'])
        plotter.plot('PCC (Valence)', 'val', 'PCC', epoch, val_metrics['PCC_valence'])
        plotter.plot('CCC (Valence)', 'val', 'CCC', epoch, val_metrics['CCC_valence'])
        plotter.plot('SAGR (Valence)', 'val', 'SAGR', epoch, val_metrics['SAGR_valence'])
        plotter.plot(f'Accuracy', 'val', 'Accuracy', epoch, val_metrics['Accuracy'])

        # Save the best model of this fold
        save_best_model(total_val_loss.avg, epoch, model, optimizer, loss_func2, os.path.join(save_dir), logger)
        scheduler.step()

        # Save the final model of this fold
    save_model(epoch, model, optimizer, loss_func2, save_dir, logger)
    del model

    # Test using the best model
    test_dataset = MoodEmo(data_root, test_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TEST_BATCH_SIZE"], shuffle=False,
                             num_workers=cfg["NUM_WORKERS"], drop_last=True)

    checkpoint = torch.load(os.path.join(save_dir, f'best_model.pth'))
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()

    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence', 'Accuracy', 'F1Score']
    test_metrics = {key: 0 for key in keys}
    target_true_mood = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_mood = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_true_valence = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_valence = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))

    with torch.no_grad():
        for test_batch_idx, test_dict in enumerate(tqdm(test_loader)):
            x_test_mood = test_dict['frames'].to(device)
            # x_test_mood_previous = test_dict['video_previous_mood_vec'].to(device)
            # x_test_delta = test_dict['video_delta_vec'].to(device)
            y_test_mood = test_dict['mood'].to(device)
            x_test_img = test_dict['current_image'].to(device)
            y_test_img = test_dict['current_valence'].flatten().to(device)
            # print('y_test shape:', y_test.shape)

            out_dict = test_model(x_test_mood, x_test_img)
            # print('test_pred shape:', out_dict['out'].ravel().to(device).shape)

            target_true_mood[test_batch_idx, :] = y_test_mood.clone().detach().cpu().numpy()
            target_pred_mood[test_batch_idx, :] = torch.max(out_dict['pred'], dim=1)[1].clone().detach().cpu().numpy()
            target_true_valence[test_batch_idx, :] = y_test_img.clone().detach().cpu().numpy()
            target_pred_valence[test_batch_idx, :] = torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy()
            # print(test_pred)

    target_true_mood = np.squeeze(np.asarray(target_true_mood)).flatten()
    target_pred_mood = np.squeeze(np.asarray(target_pred_mood)).flatten()
    target_true_valence = np.squeeze(np.asarray(target_true_valence)).flatten()
    target_pred_valence = np.squeeze(np.asarray(target_pred_valence)).flatten()

    test_metrics['RMSE_valence'] = RMSE(target_true_valence, target_pred_valence)
    test_metrics['PCC_valence'] = PCC(target_true_valence, target_pred_valence)
    test_metrics['CCC_valence'] = CCC(target_true_valence, target_pred_valence)
    test_metrics['SAGR_valence'] = SAGR(target_true_valence, target_pred_valence)
    test_metrics['Accuracy'] = ACC(target_true_mood, target_pred_mood)
    test_metrics['F1Score'] = f1_score(target_true_mood, target_pred_mood, average='weighted')

    logger.info('Test set ({:d} samples): \t'.format(len(test_loader.dataset)))
    logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}\tAcc: {:.4f}\tF1Score: {:.4f}'.format(
        test_metrics['RMSE_valence'], test_metrics['PCC_valence'], test_metrics['CCC_valence'],
        test_metrics['SAGR_valence'], test_metrics['Accuracy'], test_metrics['F1Score']))

    cm_mood = conf_mat(y_true=target_true_mood, y_pred=target_pred_mood, labels=[2, 0, 1],
                       display_labels=["Negative", "Neutral", "Positive"],
                       savefig_path=os.path.join(save_dir, f'conf_mat_mood.png'))

    return dict(train_metrics=train_metrics, train_loss=total_train_loss.avg, val_loss=total_val_loss.avg,
                y_true_mood=target_true_mood, y_pred_mood=target_pred_mood, y_true_valence=target_true_valence,
                y_pred_valence=target_pred_valence, val_metrics=val_metrics, test_metrics=test_metrics, conf_mat=cm_mood)


def teacher_student(model, data_root, train_df, val_df, test_df, cfg, logger, device, plotter,
                    save_dir, test_model):
    train_dataset = MoodEmo(data_root, train_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256,
                                augment=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, sampler=None,
                              num_workers=cfg["NUM_WORKERS"], drop_last=True)

    val_dataset = MoodEmo(data_root, val_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256,
                              augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)

    teacher = get_model(cfg["TEACHER_MODEL"], cfg, logger).to(device)
    checkpoint = torch.load(os.path.join("experiments", cfg["TEACHER_DIR"], f'best_model.pth'))
    logger.info(f'Loading teacher model')
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # print(checkpoint['model_state_dict'])
    teacher.load_state_dict(state_dict)
    teacher.eval()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg["GAMMA"])
    student_loss_func = torch.nn.MSELoss()
    distillation_loss = torch.nn.MSELoss()
    # prediction_loss = torch.nn.MSELoss()
    save_best_model = SaveBestModel()
    train_loss = AverageMeter()
    val_loss = AverageMeter()
    train_dist_loss = AverageMeter()
    train_stu_loss = AverageMeter()
    train_pred_loss = AverageMeter()
    val_dist_loss = AverageMeter()
    val_stu_loss = AverageMeter()
    val_pred_loss = AverageMeter()
    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence']
    train_metrics = {key: AverageMeter() for key in keys}
    wt = (cfg["ALPHA"], 1 - cfg["ALPHA"])
    for epoch in range(1, cfg["NUM_EPOCHS"] + 1):
        logger.info(f'Epoch {epoch} / {cfg["NUM_EPOCHS"]}')
        # Train phase
        running_train_loss = 0.0
        running_dist_loss = 0.0
        model.train()
        for tr_batch_idx, train_dict in enumerate(train_loader):
            x_train_clip = train_dict['frames'].to(device)
            x_train_img = train_dict['current_image'].to(device)
            y_train = train_dict['current_valence'].to(device)
            with torch.no_grad():
                teacher.fc[4].register_forward_hook(get_activation('feat'))
                teacher_dict = teacher(x_train_clip, x_train_img)
                teacher_feat = activation['feat']
                #print(teacher_feat.shape)
            optimizer.zero_grad()
            out_dict = model(x_train_img)
            student_loss = student_loss_func(torch.tanh(out_dict['valence'].ravel()), y_train.float())
            distill_loss = distillation_loss(out_dict['feat'], teacher_feat)
            pred_loss = dyn_wt_mse_ccc_loss(torch.tanh(out_dict['valence'].ravel()), teacher_dict['valence'], epoch=epoch,
                                             max_epochs=cfg["NUM_EPOCHS"], weight_exponent=2, clamp=False)
            loss = wt[0] * student_loss + wt[1] * distill_loss + 0.5 * pred_loss
            loss.backward()
            optimizer.step()

            RMSE_valence = RMSE(torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy(),
                                y_train.clone().detach().cpu().numpy())
            PCC_valence = PCC(torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy(),
                              y_train.clone().detach().cpu().numpy())
            CCC_valence = CCC(torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy(),
                              y_train.clone().detach().cpu().numpy())
            SAGR_valence = SAGR(torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy(),
                                y_train.clone().detach().cpu().numpy())

            train_metrics['RMSE_valence'].update(RMSE_valence, y_train.size(0))
            train_metrics['PCC_valence'].update(PCC_valence, y_train.size(0))
            train_metrics['CCC_valence'].update(CCC_valence, y_train.size(0))
            train_metrics['SAGR_valence'].update(SAGR_valence, y_train.size(0))

            # Log
            running_train_loss += student_loss.item()
            running_dist_loss += distill_loss.item()
            train_loss.update(loss.item(), y_train.size(0))
            train_dist_loss.update(distill_loss.item(), y_train.size(0))
            train_stu_loss.update(student_loss.item(), y_train.size(0))
            train_pred_loss.update(pred_loss.item(), y_train.size(0))
            if (tr_batch_idx + 1) % cfg["LOG_INTERVAL"] == 0:
                avg_loss = running_train_loss / cfg["LOG_INTERVAL"]
                avg_dist_loss = running_dist_loss / cfg["LOG_INTERVAL"]
                logger.info('[{}/{} ({:.0f}%)]\t Student Loss:{:.4f}\t Dist loss:{:.4f}\t Pred loss:{:.4f}\t Total Loss:{:.4f}'.format(
                    (tr_batch_idx + 1) * len(x_train_img), len(train_loader.dataset),
                    100 * tr_batch_idx / len(train_loader), train_stu_loss.avg, train_dist_loss.avg, train_pred_loss.avg, train_loss.avg))
                running_train_loss = 0.0

        logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            train_metrics['RMSE_valence'].avg, train_metrics['PCC_valence'].avg, train_metrics['CCC_valence'].avg,
            train_metrics['SAGR_valence'].avg))

        plotter.plot('RMSE (Valence)', 'train', 'RMSE', epoch, train_metrics['RMSE_valence'].avg)
        plotter.plot('PCC (Valence)', 'train', 'PCC', epoch, train_metrics['PCC_valence'].avg)
        plotter.plot('CCC (Valence)', 'train', 'CCC', epoch, train_metrics['CCC_valence'].avg)
        plotter.plot('SAGR (Valence)', 'train', 'SAGR', epoch, train_metrics['SAGR_valence'].avg)

        logger.info('Train set ({:d} samples): Average train loss: {:.4f}\t'.format(len(train_loader.dataset), train_loss.avg))
        plotter.plot('Loss', 'train', 'Student Loss', epoch, train_stu_loss.avg)
        plotter.plot('Loss', 'train', 'Distillation Loss', epoch, train_dist_loss.avg)
        plotter.plot('Loss', 'train', 'Prediction Loss', epoch, train_pred_loss.avg)
        # Validation phase
        model.eval()
        val_true = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence']
        val_metrics = {key: AverageMeter() for key in keys}
        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(tqdm(val_loader)):
                x_val_clip = val_dict['frames'].to(device)
                x_val_img = val_dict['current_image'].to(device)
                y_val = val_dict['mood'].to(device)
                teacher.fc[6].register_forward_hook(get_activation('feat'))
                teacher_dict = teacher(x_val_clip, x_val_img)
                teacher_feat = activation['feat']
                out_dict = model(x_val_img)
                student_loss = student_loss_func(torch.tanh(out_dict['valence'].ravel()), y_val)
                distill_loss = distillation_loss(out_dict['feat'], teacher_feat)
                pred_loss = dyn_wt_mse_ccc_loss(torch.tanh(out_dict['valence'].ravel()), teacher_dict['valence'], epoch=epoch,
                                                 max_epochs=cfg['NUM_EPOCHS'], clamp=False)
                loss = wt[0] * student_loss + wt[1] * distill_loss + 0.5 * pred_loss
                val_loss.update(loss.item(), y_val.size(0))
                val_stu_loss.update(student_loss.item(), y_val.size(0))
                val_dist_loss.update(distill_loss.item(), y_val.size(0))
                val_pred_loss.update(pred_loss.item(), y_val.size(0))
                val_true[val_batch_idx, :] = y_val.clone().detach().cpu().numpy()
                val_pred[val_batch_idx, :] = torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy()

            val_true = np.squeeze(np.asarray(val_true)).flatten()
            val_pred = np.squeeze(np.asarray(val_pred)).flatten()

        val_metrics['RMSE_valence'] = RMSE(val_true, val_pred)
        val_metrics['PCC_valence'] = PCC(val_true, val_pred)
        val_metrics['CCC_valence'] = CCC(val_true, val_pred)
        val_metrics['SAGR_valence'] = SAGR(val_true, val_pred)

        logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            val_metrics['RMSE_valence'], val_metrics['PCC_valence'], val_metrics['CCC_valence'], val_metrics['SAGR_valence']))

        logger.info('Val set ({:d} samples):'.format(len(val_loader.dataset)))
        logger.info('Validation: Student loss:{:.4f}\t Dist loss:{:.4f}\t Pred loss:{:.4f}\t Total loss:{:.4f}'.format(
            val_stu_loss.avg, val_dist_loss.avg, val_pred_loss.avg, val_loss.avg))
        plotter.plot('Loss', 'val', 'Student Loss', epoch, val_stu_loss.avg)
        plotter.plot('Loss', 'val', 'Distillation Loss', epoch, val_dist_loss.avg)
        plotter.plot('Loss', 'val', 'Prediction Loss', epoch, val_pred_loss.avg)
        plotter.plot('RMSE (Valence)', 'val', 'RMSE', epoch, val_metrics['RMSE_valence'])
        plotter.plot('PCC (Valence)', 'val', 'PCC', epoch, val_metrics['PCC_valence'])
        plotter.plot('CCC (Valence)', 'val', 'CCC', epoch, val_metrics['CCC_valence'])
        plotter.plot('SAGR (Valence)', 'val', 'SAGR', epoch, val_metrics['SAGR_valence'])

        # Save the best model of this fold
        save_best_model(val_loss.avg, epoch, model, optimizer, student_loss_func, os.path.join(save_dir), logger)
        scheduler.step()

    # Save the final model of this fold
    save_model(epoch, model, optimizer, student_loss_func, save_dir, logger)
    del model

    # Test using the best model
    test_dataset = MoodEmo(data_root, test_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256,
                              augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)
    checkpoint = torch.load(os.path.join(save_dir, f'best_model.pth'))
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()
    target_true = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence']
    test_metrics = {key: 0 for key in keys}
    with torch.no_grad():
        for test_batch_idx, test_dict in enumerate(tqdm(test_loader)):
            x_test = test_dict['current_image'].to(device)
            y_test = test_dict['current_valence'].to(device)
            out_dict = test_model(x_test)

            target_true[test_batch_idx, :] = y_test.clone().detach().cpu().numpy()
            target_pred[test_batch_idx, :] = torch.tanh(out_dict['valence'].ravel()).clone().detach().cpu().numpy()
    target_true = np.squeeze(np.asarray(target_true)).flatten()
    target_pred = np.squeeze(np.asarray(target_pred)).flatten()

    test_metrics['RMSE_valence'] = RMSE(target_true, target_pred)
    test_metrics['PCC_valence'] = PCC(target_true, target_pred)
    test_metrics['CCC_valence'] = CCC(target_true, target_pred)
    test_metrics['SAGR_valence'] = SAGR(target_true, target_pred)

    logger.info('Test set ({:d} samples): \t'.format(len(test_loader.dataset)))
    logger.info('RMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
        test_metrics['RMSE_valence'], test_metrics['PCC_valence'], test_metrics['CCC_valence'], test_metrics['SAGR_valence']))

    return dict(train_loss=train_loss.avg, val_loss=val_loss.avg, train_distillation_loss=train_dist_loss.avg,
                val_distillation_loss=val_dist_loss.avg, train_stdent_loss=train_stu_loss.avg,
                val_student_loss=val_stu_loss.avg, y_true=target_true, y_pred=target_pred, train_metrics=train_metrics,
                val_metrics=val_metrics, test_metrics=test_metrics)


# train_test_cv contains the loop for 5-fold cross validation and obtaining all the results of the 5 folds.
def train_test(model_name, data_root, train_df, val_df, test_df, cfg, logger, device, plotter, save_dir):

    logger.info(f"Train samples: {len(train_df)}")
    # logger.info(f"Train mood label distribution: {dict(Counter(train_df['mood'].to_list()))}")

    logger.info(f"Val samples: {len(val_df)}")
    # logger.info(f"Val mood label distribution: {dict(Counter(val_df['mood'].to_list()))}")

    logger.info(f"Test samples: {len(test_df)}")
    # logger.info(f"Test mood label distribution: {dict(Counter(test_df['mood'].to_list()))}")

    model = get_model(model_name, cfg, logger).to(device)

    # checkpoint = torch.load(model_path, map_location=device)
    logger.info('Training started..')
    test_model = get_model(model_name, cfg, logger).to(device)
    final_dict = train_resnet_emonet(model, data_root, train_df, val_df, test_df, cfg, logger, device, plotter, save_dir, test_model)
    return final_dict

