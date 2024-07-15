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
from affwild_cnn3d import AffWildCnn3D, AffWildMoodDelta
import torch.optim as optim
from model import get_model
from utils import AverageMeter, VisdomLinePlotter, save_model, SaveBestModel, conf_mat

torch.manual_seed(5)

# train_test performs training, validation and testing for a single fold in k-fold cross-validation
def train_test_fold(model, data_root, train_df, val_df, test_df, cfg, logger, device, plotter,
                    save_dir, test_model):
    train_dataset = AffWildMoodDelta(data_root, train_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                     width=cfg["IMAGE_SHAPE"], augment=True)
    val_dataset = AffWildMoodDelta(data_root, val_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                   width=cfg["IMAGE_SHAPE"], augment=True)

    unique_labels, counts = np.unique(train_df['mood'], return_counts=True)
    logger.info(f"Unique labels: {unique_labels}")
    logger.info(f"Counts: {counts}")

    # Assign weights to each input sample from training
    dataloader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True, sampler=None)
    y_true = []
    for tr_batch_idx, train_dict in enumerate(tqdm(dataloader)):
        y_true.append(train_dict['mood'])
    y_true = torch.flatten(torch.stack(y_true, dim=0))
    logger.info(f'Labels and count: {np.unique(y_true.numpy(), return_counts=True)}')
    # CHANGE WEIGHTS ACCORDINGLY (VERY IMPORTANT)
    weights = torch.tensor([0.125, 0.625, 1.0], dtype=torch.float)
    samples_weights = weights[y_true]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, sampler=sampler,
                              num_workers=cfg["NUM_WORKERS"], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg["GAMMA"])
    # tr_loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 5., 10.]).to(device))
    save_best_model = SaveBestModel()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    for epoch in range(1, cfg["NUM_EPOCHS"]+1):
        logger.info(f'Epoch {epoch} / {cfg["NUM_EPOCHS"]}')
        # Train phase
        running_train_loss = 0.0
        model.train()
        for tr_batch_idx, train_dict in enumerate(train_loader):
            x_train = train_dict['frames'].to(device)
            y_train = train_dict['mood'].to(device)
            optimizer.zero_grad()
            out_dict = model(x_train)
            loss = loss_func(out_dict['out'], y_train)
            loss.backward()
            optimizer.step()
            tr_pred = torch.max(out_dict['out'], dim=1)[1]
            train_correct = (tr_pred == y_train).sum()

            # Log
            running_train_loss += loss.item()
            train_loss.update(loss.item(), y_train.size(0))
            train_accuracy.update(train_correct.item() / y_train.size(0), y_train.size(0))
            if (tr_batch_idx + 1) % cfg["LOG_INTERVAL"] == 0:
                avg_loss = running_train_loss / cfg["LOG_INTERVAL"]
                logger.info('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    (tr_batch_idx + 1) * len(x_train), len(train_loader.dataset),
                    100 * tr_batch_idx / len(train_loader),
                    avg_loss, train_accuracy.avg * 100))
                running_train_loss = 0.0
        logger.info('Train set ({:d} samples): Average train loss: {:.4f}\tTrain Accuracy: {:.4f}'.format(
            len(train_loader.dataset), train_loss.avg, train_accuracy.avg))
        plotter.plot(f'Accuracy', 'train', 'Accuracy', epoch, train_accuracy.avg)
        plotter.plot(f'Loss', 'train', 'Total Loss', epoch, train_loss.avg)
        # Validation phase
        model.eval()
        val_true_vec = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_vec = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(tqdm(val_loader)):
                x_val = val_dict['frames'].to(device)
                y_val = val_dict['mood'].to(device)
                out_dict = model(x_val)
                loss = loss_func(out_dict['out'], y_val)
                val_pred = torch.max(out_dict['out'], dim=1)[1]
                val_correct = (val_pred == y_val).sum()
                val_loss.update(loss.item(), y_val.size(0))
                val_accuracy.update(val_correct.item() / y_val.size(0), y_val.size(0))
                val_true_vec[val_batch_idx, :] = y_val.clone().detach().cpu().numpy()
                val_pred_vec[val_batch_idx, :] = val_pred.detach().cpu().numpy()
        val_true_vec = np.squeeze(np.asarray(val_true_vec)).flatten()
        val_pred_vec = np.squeeze(np.asarray(val_pred_vec)).flatten()
        f1_weighted = f1_score(val_true_vec, val_pred_vec, average='weighted')

        # cm = confusion_matrix(target_true, target_pred, labels=[2, 0, 1])
        # cm = cm / cm.sum(axis=1, keepdims=True)
        # val_cm = conf_mat(y_true=val_true_vec, y_pred=val_pred_vec, labels=[2, 0, 1],
        #               display_labels=["Negative", "Neutral", "Positive"],
        #               #savefig_path=os.path.join(save_dir, f'conf_mat.png')
        #               )
        # logger.info(f"Confusion matrix for epoch {epoch}: \n {val_cm}")
        logger.info('Val set ({:d} samples): Average val loss: {:.4f}\tVal Accuracy: {:.4f}'.format(
            len(val_loader.dataset), val_loss.avg, val_accuracy.avg))
        plotter.plot(f'Accuracy', 'val', 'Accuracy', epoch, val_accuracy.avg)
        plotter.plot(f'Loss', 'val', 'Total Loss', epoch, val_loss.avg)

        # Save the best model of this fold
        save_best_model(val_loss.avg, epoch, model, optimizer, loss_func, os.path.join(save_dir), logger)
        scheduler.step()

    # Save the final model of this fold
    save_model(epoch, model, optimizer, loss_func, save_dir, logger)
    del model

    # Test using the best model
    test_dataset = AffWildMoodDelta(data_root, test_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                    width=cfg["IMAGE_SHAPE"], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TEST_BATCH_SIZE"], shuffle=False,
                             num_workers=cfg["NUM_WORKERS"], drop_last=True)

    checkpoint = torch.load(os.path.join(save_dir, f'best_model.pth'))
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()
    target_true = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    test_accuracy = AverageMeter()
    cm_list = []
    with torch.no_grad():
        for test_batch_idx, test_dict in enumerate(tqdm(test_loader)):
            x_test = test_dict['frames'].to(device)
            y_test = test_dict['mood'].to(device)
            out_dict = test_model(x_test)
            test_pred = torch.max(out_dict['out'], dim=1)[1]
            test_correct = (test_pred == y_test).sum()
            test_accuracy.update(test_correct.item() / y_test.size(0), y_test.size(0))
            target_true[test_batch_idx, :] = y_test.clone().detach().cpu().numpy()
            target_pred[test_batch_idx, :] = test_pred.detach().cpu().numpy()
    target_true = np.squeeze(np.asarray(target_true)).flatten()
    target_pred = np.squeeze(np.asarray(target_pred)).flatten()
    f1_weighted = f1_score(target_true, target_pred, average='weighted')
    logger.info('Test set ({:d} samples): \tTest Accuracy: {:.4f}'
                ' \tWtd f-score: {:.4f}'.format(len(test_loader.dataset), test_accuracy.avg, f1_weighted))
    # cm = confusion_matrix(target_true, target_pred, labels=[2, 0, 1])
    # cm = cm / cm.sum(axis=1, keepdims=True)
    cm = conf_mat(y_true=target_true, y_pred=target_pred, labels=[2, 0, 1],
                  display_labels=["Negative", "Neutral", "Positive"],
                  savefig_path=os.path.join(save_dir, f'conf_mat.png'))
    return dict(train_accuracy=train_accuracy.avg, val_accuracy=val_accuracy.avg, test_accuracy=test_accuracy.avg,
                wtd_fscore=f1_weighted, train_loss=train_loss.avg, val_loss=val_loss.avg, y_true=target_true,
                y_pred=target_pred, conf_mat=cm)


# Use while training two branch model with mood and delta labels.
def train_test_fold_twobranch(model, data_root, train_df, val_df, test_frames_df, cfg, logger, device, plotter,
                              save_dir, test_model):
    train_dataset = AffWildMoodDelta(data_root, train_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                     width=cfg["IMAGE_SHAPE"], augment=True)
    val_dataset = AffWildMoodDelta(data_root, val_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                   width=cfg["IMAGE_SHAPE"], augment=False)

    unique_labels, counts = np.unique(train_df['mood'], return_counts=True)
    logger.info(f"Unique labels: {unique_labels}")
    logger.info(f"Counts: {counts}")

    # Assign weights to each input sample from training
    dataloader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True, sampler=None)
    y_true = []
    for tr_batch_idx, train_dict in enumerate(tqdm(dataloader)):
        y_true.append(train_dict['mood'])
    y_true = torch.flatten(torch.stack(y_true, dim=0))
    logger.info(f'Labels and count: {np.unique(y_true.numpy(), return_counts=True)}')
    # CHANGE WEIGHTS ACCORDINGLY (VERY IMPORTANT)
    weights = torch.tensor([0.125, 0.625, 1.0], dtype=torch.float)
    samples_weights = weights[y_true]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, sampler=sampler,
                              num_workers=cfg["NUM_WORKERS"], drop_last=True)
    y_true = []
    for tr_batch_idx, train_dict in enumerate(tqdm(train_loader)):
        y_true.append(train_dict['mood'])
    y_true = torch.flatten(torch.stack(y_true, dim=0))
    logger.info(f'Labels and count after sampler: {np.unique(y_true.numpy(), return_counts=True)}')

    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg["GAMMA"])
    # tr_loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 5., 10.]).to(device))
    loss_func2 = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 3.25]).to(device))
    save_best_model = SaveBestModel()
    train_loss_mood = AverageMeter()
    train_loss_delta = AverageMeter()
    val_loss_mood = AverageMeter()
    val_loss_delta = AverageMeter()
    overall_train_loss = AverageMeter()
    overall_val_loss = AverageMeter()
    train_accuracy_mood = AverageMeter()
    train_accuracy_delta = AverageMeter()
    val_accuracy_mood = AverageMeter()
    val_accuracy_delta = AverageMeter()
    for epoch in range(1, cfg["NUM_EPOCHS"]+1):
        logger.info(f'Epoch {epoch} / {cfg["NUM_EPOCHS"]}')
        # Train phase
        running_train_loss_mood = 0.0
        running_train_loss_delta = 0.0
        running_train_overall_loss = 0.0
        model.train()
        for tr_batch_idx, train_dict in enumerate(train_loader):
            x_train = train_dict['frames'].to(device)
            y_train = train_dict['mood'].to(device)
            y_train_delta = train_dict['delta_expression'].to(device)
            # print(np.unique(y_train_delta.clone().cpu().numpy()))
            # print(y_train.shape)
            optimizer.zero_grad()
            out_dict = model(x_train)
            loss1 = loss_func1(out_dict['out1'], y_train)
            loss2 = loss_func2(out_dict['out2'], y_train_delta)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            tr_pred_mood = torch.max(out_dict['out1'], dim=1)[1]
            tr_pred_delta = torch.max(out_dict['out2'], dim=1)[1]
            # print(tr_pred)
            train_correct_mood = (tr_pred_mood == y_train).sum()
            train_correct_delta = (tr_pred_delta == y_train_delta).sum()

            # Log
            running_train_overall_loss += loss.item()
            overall_train_loss.update(loss.item(), y_train.size(0))
            running_train_loss_mood += loss1.item()
            train_loss_mood.update(loss1.item(), y_train.size(0))
            running_train_loss_delta += loss2.item()
            train_loss_delta.update(loss2.item(), y_train_delta.size(0))
            train_accuracy_mood.update(train_correct_mood.item() / y_train.size(0), y_train.size(0))
            train_accuracy_delta.update(train_correct_delta.item() / y_train_delta.size(0), y_train_delta.size(0))
            if (tr_batch_idx + 1) % cfg["LOG_INTERVAL"] == 0:
                avg_loss_mood = running_train_loss_mood / cfg["LOG_INTERVAL"]
                avg_loss_delta = running_train_loss_delta / cfg["LOG_INTERVAL"]
                logger.info('[{}/{} ({:.0f}%)]\t Mood Loss: {:.6f}\t Mood Accuracy:{:.3f}%'.format(
                    (tr_batch_idx + 1) * len(x_train), len(train_loader.dataset),
                    100 * tr_batch_idx / len(train_loader),
                    avg_loss_mood, train_accuracy_mood.avg * 100))
                logger.info('[{}/{} ({:.0f}%)]\t Delta Loss: {:.6f}\t Delta Accuracy:{:.3f}%'.format(
                    (tr_batch_idx + 1) * len(x_train), len(train_loader.dataset),
                    100 * tr_batch_idx / len(train_loader),
                    avg_loss_delta, train_accuracy_delta.avg * 100))
                running_train_loss_mood = 0.0
                running_train_loss_delta = 0.0
                running_train_overall_loss = 0.0
        logger.info('Train set ({:d} samples): Average train loss mood: {:.4f}\tTrain accuracy mood: {:.4f}'.format(
            len(train_loader.dataset), train_loss_mood.avg, train_accuracy_mood.avg))
        logger.info('Train set ({:d} samples): Average train loss delta: {:.4f}\tTrain accuracy delta: {:.4f}'.format(
            len(train_loader.dataset), train_loss_delta.avg, train_accuracy_delta.avg))
        plotter.plot(f'Mood Accuracy', 'train', 'Accuracy', epoch, train_accuracy_mood.avg)
        plotter.plot(f'Mood Loss', 'train', 'Total Loss', epoch, train_loss_mood.avg)
        plotter.plot(f'Delta Accuracy', 'train', 'Accuracy', epoch, train_accuracy_delta.avg)
        plotter.plot(f'Delta Loss', 'train', 'Total Loss', epoch, train_loss_delta.avg)
        # Validation phase
        model.eval()

        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(tqdm(val_loader)):
                x_val = val_dict['frames'].to(device)
                y_val = val_dict['mood'].to(device)
                y_val_delta = val_dict['delta_expression'].to(device)
                out_dict = model(x_val)
                loss1 = loss_func1(out_dict['out1'], y_val)
                loss2 = loss_func2(out_dict['out2'], y_val_delta)
                loss = loss1 + loss2
                val_pred_mood = torch.max(out_dict['out1'], dim=1)[1]
                val_pred_delta = torch.max(out_dict['out2'], dim=1)[1]
                val_correct_mood = (val_pred_mood == y_val).sum()
                val_correct_delta = (val_pred_delta == y_val_delta).sum()
                val_loss_mood.update(loss1.item(), y_val.size(0))
                val_loss_delta.update(loss2.item(), y_val_delta.size(0))
                overall_val_loss.update(loss.item(), y_val.size(0))
                val_accuracy_mood.update(val_correct_mood.item() / y_val.size(0), y_val.size(0))
                val_accuracy_delta.update(val_correct_delta.item() / y_val_delta.size(0), y_val_delta.size(0))
        logger.info('Val set ({:d} samples): Average val loss mood: {:.4f}\tVal Accuracy mood: {:.4f}'.format(
            len(val_loader.dataset), val_loss_mood.avg, val_accuracy_mood.avg))
        logger.info('Val set ({:d} samples): Average val loss delta: {:.4f}\tVal Accuracy delta: {:.4f}'.format(
            len(val_loader.dataset), val_loss_delta.avg, val_accuracy_delta.avg))
        plotter.plot(f'Mood Accuracy', 'val', 'Mood Accuracy', epoch, val_accuracy_mood.avg)
        plotter.plot(f'Mood Loss', 'val', 'Mood Loss', epoch, val_loss_mood.avg)
        plotter.plot(f'Delta Accuracy', 'val', 'Delta Accuracy', epoch, val_accuracy_delta.avg)
        plotter.plot(f'Delta Loss', 'val', 'Delta Loss', epoch, val_loss_delta.avg)

        # Save the best model of this fold
        save_best_model(val_loss_mood.avg, epoch, model, optimizer, loss_func1, os.path.join(save_dir), logger)
        scheduler.step()

    # Save the final model of this fold
    save_model(epoch, model, optimizer, loss_func1, save_dir, logger)
    del model

    # Test using the best model
    test_dataset = AffWildMoodDelta(data_root, test_frames_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                    width=cfg["IMAGE_SHAPE"], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TEST_BATCH_SIZE"], shuffle=False,
                             num_workers=cfg["NUM_WORKERS"], drop_last=True)

    checkpoint = torch.load(os.path.join(save_dir, f'best_model.pth'))
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()
    target_true_mood = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_mood = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_true_delta = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred_delta = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    test_accuracy_mood = AverageMeter()
    test_accuracy_delta = AverageMeter()
    with torch.no_grad():
        for test_batch_idx, test_dict in enumerate(tqdm(test_loader)):
            x_test = test_dict['frames'].to(device)
            y_test = test_dict['mood'].to(device)
            y_test_delta = test_dict['delta_expression'].to(device)
            # print(y_test.shape)
            out_dict = test_model(x_test)
            # print(out.data)
            test_pred_mood = torch.max(out_dict['out1'], dim=1)[1]
            test_pred_delta = torch.max(out_dict['out2'], dim=1)[1]
            # test_pred = torch.max(out.data, dim=1)
            # print(test_pred)
            test_correct_mood = (test_pred_mood == y_test).sum()
            test_accuracy_mood.update(test_correct_mood.item() / y_test.size(0), y_test.size(0))
            test_correct_delta = (test_pred_delta == y_test_delta).sum()
            test_accuracy_delta.update(test_correct_delta.item() / y_test_delta.size(0), y_test_delta.size(0))
            target_true_mood[test_batch_idx, :] = y_test.clone().detach().cpu().numpy()
            target_pred_mood[test_batch_idx, :] = test_pred_mood.detach().cpu().numpy()
            target_true_delta[test_batch_idx, :] = y_test_delta.clone().detach().cpu().numpy()
            target_pred_delta[test_batch_idx, :] = test_pred_delta.detach().cpu().numpy()
    target_true_mood = np.squeeze(np.asarray(target_true_mood)).flatten()
    target_pred_mood = np.squeeze(np.asarray(target_pred_mood)).flatten()
    f1_weighted_mood = f1_score(target_true_mood, target_pred_mood, average='weighted')
    target_true_delta = np.squeeze(np.asarray(target_true_delta)).flatten()
    target_pred_delta = np.squeeze(np.asarray(target_pred_delta)).flatten()
    f1_weighted_delta = f1_score(target_true_delta, target_pred_delta, average='weighted')
    logger.info('Test set ({:d} samples): \tTest Accuracy Mood: {:.4f}'
                ' \tWtd f-score Mood: {:.4f}'.format(len(test_loader.dataset), test_accuracy_mood.avg, f1_weighted_mood))
    logger.info('Test set ({:d} samples): \tTest Accuracy Delta: {:.4f}'
                ' \tWtd f-score Delta: {:.4f}'.format(len(test_loader.dataset), test_accuracy_delta.avg,
                                                      f1_weighted_delta))
    # cm = confusion_matrix(target_true, target_pred, labels=[2, 0, 1])
    # cm = cm / cm.sum(axis=1, keepdims=True)
    cm_mood = conf_mat(y_true=target_true_mood, y_pred=target_pred_mood, labels=[2, 0, 1],
                       display_labels=["Negative", "Neutral", "Positive"],
                       savefig_path=os.path.join(save_dir, f'conf_mat_mood.png'))
    cm_delta = conf_mat(y_true=target_true_delta, y_pred=target_pred_delta, labels=[0, 1],
                        display_labels=["Dissimilar", "Similar"],
                        savefig_path=os.path.join(save_dir, f'conf_mat_delta.png'))

    return dict(train_accuracy_mood=train_accuracy_mood.avg, val_accuracy_mood=val_accuracy_mood.avg,
                train_accuracy_delta=train_accuracy_delta.avg, val_accuracy_delta=val_accuracy_delta.avg,
                test_accuracy_mood=test_accuracy_mood.avg, wtd_fscore_mood=f1_weighted_mood,
                test_accuracy_delta=test_accuracy_delta.avg, wtd_fscore_delta=f1_weighted_delta,
                train_loss_mood=train_loss_mood.avg, val_loss_mood=val_loss_mood.avg,
                train_loss_delta=train_loss_delta.avg, val_loss_delta=val_loss_delta.avg,
                overall_train_loss=overall_train_loss.avg, overall_val_loss=overall_val_loss.avg,
                y_true_mood=target_true_mood, y_pred_mood=target_pred_mood, conf_mat_mood=cm_mood,
                y_pred_delta=target_pred_delta, y_true_delta=target_true_delta, conf_mat_delta=cm_delta)


def teacher_student(model, data_root, train_df, val_df, test_df, cfg, logger, device, plotter,
                    save_dir, test_model):
    train_dataset = AffWildMoodDelta(data_root, train_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                     width=cfg["IMAGE_SHAPE"], augment=True)
    val_dataset = AffWildMoodDelta(data_root, val_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                   width=cfg["IMAGE_SHAPE"], augment=True)

    unique_labels, counts = np.unique(train_df['mood'], return_counts=True)
    logger.info(f"Unique labels: {unique_labels}")
    logger.info(f"Counts: {counts}")

    # Assign weights to each input sample from training
    dataloader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True, sampler=None)
    y_true = []
    for tr_batch_idx, train_dict in enumerate(tqdm(dataloader)):
        y_true.append(train_dict['mood'])
    y_true = torch.flatten(torch.stack(y_true, dim=0))
    logger.info(f'Labels and count: {np.unique(y_true.numpy(), return_counts=True)}')
    # CHANGE WEIGHTS ACCORDINGLY (VERY IMPORTANT)
    weights = torch.tensor([0.00001, 0.00003, 0.0001], dtype=torch.float)
    samples_weights = weights[y_true]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, sampler=sampler,
                              num_workers=cfg["NUM_WORKERS"], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"],
                            drop_last=True)

    teacher = get_model(cfg["TEACHER_MODEL"], cfg, logger).to(device)
    checkpoint = torch.load(os.path.join("experiments", cfg["TEACHER_DIR"], f'best_model.pth'))
    logger.info(f'Loading teacher model')
    teacher.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg["GAMMA"])
    # tr_loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    student_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1., 10.]).to(device))
    distillation_loss = torch.nn.KLDivLoss()
    save_best_model = SaveBestModel()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_dist_loss = AverageMeter()
    train_stu_loss = AverageMeter()
    val_dist_loss = AverageMeter()
    val_stu_loss = AverageMeter()
    for epoch in range(1, cfg["NUM_EPOCHS"]+1):
        logger.info(f'Epoch {epoch} / {cfg["NUM_EPOCHS"]}')
        # Train phase
        running_train_loss = 0.0
        running_dist_loss = 0.0
        model.train()
        for tr_batch_idx, train_dict in enumerate(train_loader):
            x_train = train_dict['frames'].to(device)
            y_train = train_dict['mood'].to(device)
            with torch.no_grad():
                teacher_dict = teacher(x_train)
            optimizer.zero_grad()
            out_dict = model(x_train)
            student_loss = student_loss_func(out_dict['out'], y_train)
            tr_pred = torch.max(out_dict['out'], dim=1)[1]
            train_correct = (tr_pred == y_train).sum()
            distill_loss = distillation_loss(F.log_softmax(out_dict['out']/cfg["TEMPERATURE"], dim=1),
                                             F.softmax(teacher_dict['out1']/cfg["TEMPERATURE"], dim=1))
            loss = cfg["ALPHA"] * student_loss + (1 - cfg["ALPHA"]) * distill_loss
            loss.backward()
            optimizer.step()

            # Log
            running_train_loss += student_loss.item()
            running_dist_loss += distill_loss.item()
            train_loss.update(loss.item(), y_train.size(0))
            train_accuracy.update(train_correct.item() / y_train.size(0), y_train.size(0))
            train_dist_loss.update(distill_loss.item(), y_train.size(0))
            train_stu_loss.update(student_loss.item(), y_train.size(0))
            if (tr_batch_idx + 1) % cfg["LOG_INTERVAL"] == 0:
                avg_loss = running_train_loss / cfg["LOG_INTERVAL"]
                avg_dist_loss = running_dist_loss / cfg["LOG_INTERVAL"]
                logger.info('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    (tr_batch_idx + 1) * len(x_train), len(train_loader.dataset),
                    100 * tr_batch_idx / len(train_loader),
                    avg_loss, train_accuracy.avg * 100))
                logger.info('[{}/{} ({:.0f}%)]\t Dist loss:{:.3f}'.format(
                    (tr_batch_idx + 1) * len(x_train), len(train_loader.dataset),
                    100 * tr_batch_idx / len(train_loader), train_dist_loss.avg))
                running_train_loss = 0.0
        logger.info('Train set ({:d} samples): Average train loss: {:.4f}\tTrain Accuracy: {:.4f}'.format(
            len(train_loader.dataset), train_loss.avg, train_accuracy.avg))
        plotter.plot(f'Accuracy', 'train', 'Accuracy', epoch, train_accuracy.avg)
        plotter.plot(f'Loss', 'train', 'Student Loss', epoch, train_stu_loss.avg)
        plotter.plot(f'Loss', 'train', 'Distillation Loss', epoch, train_dist_loss.avg)
        # Validation phase
        model.eval()
        val_true_vec = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        val_pred_vec = np.zeros(shape=(len(val_loader), cfg["BATCH_SIZE"]))
        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(tqdm(val_loader)):
                x_val = val_dict['frames'].to(device)
                y_val = val_dict['mood'].to(device)
                teacher_dict = teacher(x_val)
                out_dict = model(x_val)
                student_loss = student_loss_func(out_dict['out'], y_val)
                val_pred = torch.max(out_dict['out'], dim=1)[1]
                distill_loss = distillation_loss(F.log_softmax(out_dict['out']/cfg["TEMPERATURE"], dim=1),
                                                 F.softmax(teacher_dict['out1']/cfg["TEMPERATURE"], dim=1))
                val_correct = (val_pred == y_val).sum()
                val_loss.update(loss.item(), y_val.size(0))
                val_accuracy.update(val_correct.item() / y_val.size(0), y_val.size(0))
                val_stu_loss.update(student_loss.item(), y_val.size(0))
                val_dist_loss.update(distill_loss.item(), y_val.size(0))
                val_true_vec[val_batch_idx, :] = y_val.clone().detach().cpu().numpy()
                val_pred_vec[val_batch_idx, :] = val_pred.detach().cpu().numpy()
        val_true_vec = np.squeeze(np.asarray(val_true_vec)).flatten()
        val_pred_vec = np.squeeze(np.asarray(val_pred_vec)).flatten()
        f1_weighted = f1_score(val_true_vec, val_pred_vec, average='weighted')

        # cm = confusion_matrix(target_true, target_pred, labels=[2, 0, 1])
        # cm = cm / cm.sum(axis=1, keepdims=True)
        # val_cm = conf_mat(y_true=val_true_vec, y_pred=val_pred_vec, labels=[2, 0, 1],
        #               display_labels=["Negative", "Neutral", "Positive"],
        #               #savefig_path=os.path.join(save_dir, f'conf_mat.png')
        #               )
        # logger.info(f"Confusion matrix for epoch {epoch}: \n {val_cm}")
        logger.info('Val set ({:d} samples): Average val loss: {:.4f}\tVal Accuracy: {:.4f}'.format(
            len(val_loader.dataset), val_loss.avg, val_accuracy.avg))
        plotter.plot(f'Accuracy', 'val', 'Accuracy', epoch, val_accuracy.avg)
        plotter.plot(f'Loss', 'val', 'Student Loss', epoch, val_stu_loss.avg)
        plotter.plot(f'Loss', 'val', 'Distillation Loss', epoch, val_dist_loss.avg)

        # Save the best model of this fold
        save_best_model(val_loss.avg, epoch, model, optimizer, student_loss_func, os.path.join(save_dir), logger)
        scheduler.step()

    # Save the final model of this fold
    save_model(epoch, model, optimizer, student_loss_func, save_dir, logger)
    del model

    # Test using the best model
    test_dataset = AffWildMoodDelta(data_root, test_df, cfg, logger, height=cfg["IMAGE_SHAPE"],
                                    width=cfg["IMAGE_SHAPE"], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TEST_BATCH_SIZE"], shuffle=False,
                             num_workers=cfg["NUM_WORKERS"], drop_last=True)

    checkpoint = torch.load(os.path.join(save_dir, f'best_model.pth'))
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()
    target_true = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    target_pred = np.zeros(shape=(len(test_loader), cfg["TEST_BATCH_SIZE"]))
    test_accuracy = AverageMeter()
    with torch.no_grad():
        for test_batch_idx, test_dict in enumerate(tqdm(test_loader)):
            x_test = test_dict['frames'].to(device)
            y_test = test_dict['mood'].to(device)
            out_dict = test_model(x_test)
            test_pred = torch.max(out_dict['out'], dim=1)[1]
            test_correct = (test_pred == y_test).sum()
            test_accuracy.update(test_correct.item() / y_test.size(0), y_test.size(0))
            target_true[test_batch_idx, :] = y_test.clone().detach().cpu().numpy()
            target_pred[test_batch_idx, :] = test_pred.detach().cpu().numpy()
    target_true = np.squeeze(np.asarray(target_true)).flatten()
    target_pred = np.squeeze(np.asarray(target_pred)).flatten()
    f1_weighted = f1_score(target_true, target_pred, average='weighted')
    logger.info('Test set ({:d} samples): \tTest Accuracy: {:.4f}'
                ' \tWtd f-score: {:.4f}'.format(len(test_loader.dataset), test_accuracy.avg, f1_weighted))
    # cm = confusion_matrix(target_true, target_pred, labels=[2, 0, 1])
    # cm = cm / cm.sum(axis=1, keepdims=True)
    cm = conf_mat(y_true=target_true, y_pred=target_pred, labels=[2, 0, 1],
                  display_labels=["Negative", "Neutral", "Positive"],
                  savefig_path=os.path.join(save_dir, f'conf_mat.png'))
    return dict(train_accuracy=train_accuracy.avg, val_accuracy=val_accuracy.avg, test_accuracy=test_accuracy.avg,
                wtd_fscore=f1_weighted, train_loss=train_loss.avg, val_loss=val_loss.avg,
                train_distillation_loss=train_dist_loss.avg, val_distillation_loss=val_dist_loss.avg,
                train_stdent_loss=train_stu_loss.avg, val_student_loss=val_stu_loss.avg, y_true=target_true,
                y_pred=target_pred, conf_mat=cm)


# train_test_cv contains the loop for 5-fold cross validation and obtaining all the results of the 5 folds.
def train_test(model_name, data_root, save_dir, train_frames_df, test_frames_df, logger,
               cfg, device, plotter):
    kfold = GroupKFold(n_splits=cfg["FOLDS"])
    train_idx_sub, val_idx = next(kfold.split(train_frames_df,
                                              groups=list(train_frames_df['video_id'])
                                              ))

    train_df = train_frames_df.iloc[train_idx_sub]
    train_df.reset_index(inplace=True)

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Train mood label distribution: {dict(Counter(train_df['mood'].to_list()))}")
    val_df = train_frames_df.iloc[val_idx]
    val_df.reset_index(inplace=True)

    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Val mood label distribution: {dict(Counter(val_df['mood'].to_list()))}")
    logger.info(f"Test samples: {len(test_frames_df)}")
    logger.info(f"Test mood label distribution: {dict(Counter(test_frames_df['mood'].to_list()))}")

    model = get_model(model_name, cfg, logger).to(device)

    # checkpoint = torch.load(model_path, map_location=device)
    logger.info('Training started..')
    test_model = get_model(model_name, cfg, logger).to(device)
    final_dict = teacher_student(model, data_root, train_df, val_df, test_frames_df, cfg, logger, device,
                                 plotter, save_dir, test_model)

    # conf_mat(cm_list, savefig_path=os.path.join(save_dir, f'conf_mat.png'))
    return final_dict
