from __future__ import print_function
from linearnet import LinearNetEmoMoodDelta, LinearNetVectors
from resnet_emonet import ResnetEmonet, MoodDeltaEmonet, TSNet


def get_model(name, cfg, logger):
    if name == "EmoMoodNet":
        model = LinearNetEmoMoodDelta(is_pretrained_mood=cfg['IS_PRETRAINED_MOOD'], is_pretrained_delta=cfg['IS_PRETRAINED_DELTA'],
                                      is_pretrained_emofan=cfg['IS_PRETRAINED_EMOFAN'], feat_fusion=cfg['FEAT_FUSION'],
                                      num_neurons_fc=cfg['NUM_NEURONS_FC'], dropout_rate=cfg["DROPOUT_RATE"], cfg=cfg)
        logger.info(f'Building model: \n{name}\n, '
                    f'Dropout rate: {cfg["DROPOUT_RATE"]}\n'
                    f'Pretrained mood model: {cfg["IS_PRETRAINED_MOOD"]}\n'
                    f'Pretrained delta model: {cfg["IS_PRETRAINED_DELTA"]}\n'
                    f'Pretrained EmoFAN model: {cfg["IS_PRETRAINED_EMOFAN"]}\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f"{sum(p.numel() for p in model.parameters()) / 1e6} M total parameters")
        logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M trainable parameters")
        for name, param in model.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')

    elif name == "EmoMoodVectors":
        model = LinearNetVectors(feat_fusion=cfg['FEAT_FUSION'], num_neurons_fc=cfg['NUM_NEURONS_FC'],
                                 dropout_rate=cfg["DROPOUT_RATE"])
        logger.info(f'Building model: \n{name}\n, '
                    f'Dropout rate: {cfg["DROPOUT_RATE"]}\n'
                    f'Feature fusion type: {cfg["FEAT_FUSION"]}\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f"{sum(p.numel() for p in model.parameters()) / 1e6} M total parameters")
        logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M trainable parameters")
        for name, param in model.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')

    elif name == "MoodEmoNet":
        model = ResnetEmonet(feat_fusion=cfg['FEAT_FUSION'], dropout_rate=cfg["DROPOUT_RATE"],
                             is_pretrained=cfg["IS_PRETRAINED_EMOFAN"], num_neurons_fc=cfg["NUM_NEURONS_FC"],
                             num_mood_classes=cfg["NUM_MOOD_CLASSES"], cfg=cfg)
        logger.info(f'Building model: \n{name}\n, '
                    f'Dropout rate: {cfg["DROPOUT_RATE"]}\n'
                    f'Feature fusion type: {cfg["FEAT_FUSION"]}\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f"{sum(p.numel() for p in model.parameters()) / 1e6} M total parameters")
        logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M trainable parameters")
        # for name, param in model.named_parameters():
        #     logger.info(f'{name}: {param.requires_grad}')

    elif name == "MoodDeltaEmoNet":
        model = MoodDeltaEmonet(feat_fusion=cfg['FEAT_FUSION'], dropout_rate=cfg["DROPOUT_RATE"],
                                is_pretrained=cfg["IS_PRETRAINED_EMOFAN"], num_neurons_fc=cfg["NUM_NEURONS_FC"],
                                num_mood_classes=cfg["NUM_MOOD_CLASSES"], num_delta_classes=cfg["NUM_DELTA_CLASSES"],
                                cfg=cfg)
        logger.info(f'Building model: \n{name}\n, '
                    f'Dropout rate: {cfg["DROPOUT_RATE"]}\n'
                    f'Feature fusion type: {cfg["FEAT_FUSION"]}\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f"{sum(p.numel() for p in model.parameters()) / 1e6} M total parameters")
        logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M trainable parameters")
        # for name, param in model.named_parameters():
        #    logger.info(f'{name}: {param.requires_grad}')

    elif name == "TSNet":
        model = TSNet(dropout_rate=cfg["DROPOUT_RATE"], is_pretrained=cfg["IS_PRETRAINED_EMOFAN"], num_neurons_fc=cfg["NUM_NEURONS_FC"], cfg=cfg)
        logger.info(f'Building model: \n{name}\n, '
                    f'Dropout rate: {cfg["DROPOUT_RATE"]}\n'
                    f'Feature fusion type: {cfg["FEAT_FUSION"]}\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f"{sum(p.numel() for p in model.parameters()) / 1e6} M total parameters")
        logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M trainable parameters")
        # for name, param in model.named_parameters():
        #    logger.info(f'{name}: {param.requires_grad}')

    else:
        raise NameError('Please specify current argument for name.')

    if cfg["IS_DATA_PARALLEL"]:
        from torch.nn import DataParallel
        model = DataParallel(model, device_ids=cfg["GPU_IDS"])

    return model