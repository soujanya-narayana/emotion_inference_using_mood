import math
from functools import partial
from pathlib import Path
from utils import load_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import *
from emonet import *

torch.manual_seed(5)


class ResnetEmonet(nn.Module):
    """
        Encoder for mood: ResNet
        Encoder for emotion = EmoFAN
    """

    def __init__(self, feat_fusion, dropout_rate, is_pretrained, num_neurons_fc, num_mood_classes, cfg):
        super(ResnetEmonet, self).__init__()

        # Fusion technique
        self.feat_fusion = feat_fusion

        self.dropout_rate = dropout_rate

        self.num_neurons_fc = num_neurons_fc

        self.num_mood_classes = num_mood_classes

        self.is_pretrained = is_pretrained

        self.resnet = generate_model_branchedlinear(model_depth=cfg["RESNET_DEPTH"]).to(cfg["DEVICE"])
        # print(self.resnet)

        self.emofan = EmoNet(num_modules=2, n_expression=cfg["EMOFAN_EMOTIONS"], n_reg=2, n_blocks=4, attention=True,
                             temporal_smoothing=False).to(cfg["DEVICE"])
        # print(self.emofan)

        if self.is_pretrained:
            # Load weights to emonet
            state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{cfg["EMOFAN_EMOTIONS"]}.pth')
            print(f'Loading EmoFAN from {state_dict_path}')
            state_dict = torch.load(str(state_dict_path))
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.emofan.load_state_dict(state_dict, strict=False)

        # model.emo_fc_2[3] = nn.Linear(in_features=128, out_features=n_reg, bias=True)

        self.fc_in_features = self.emofan.emo_fc_2[0].in_features  # [bs, 256]

        # get features before the last linear block
        # Refer emonet supplementary material for more info
        self.emofan.emo_fc_2 = nn.Identity()    # Returns [bs, 256]

        # either consider mood and emotion features or mood, delta and emotion features
        assert self.feat_fusion in ['concat', 'multiply']

        if self.feat_fusion == "concat":
            feature_coeff = 2
        elif self.feat_fusion == "multiply":
            feature_coeff = 1
        else:
            raise ValueError

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * feature_coeff, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, num_mood_classes + num_neurons_fc, bias=False)
        )
        self.fc.apply(self.init_weights)

        # self.max_pool = nn.MaxPool3d(1, 1)
        # self.avg_pool = nn.AvgPool3d(1, 1)
        # self.att_conv = nn.Conv3d(6, 5, kernel_size=3, padding='same')

    def forward_emonet(self, x):
        output = self.emofan(x)
        output = output['feature'].view(output['feature'].size()[0], -1)  # [bs, 256]
        return output

    def forward_resnet(self, x):
        output = self.resnet(x)
        return output['out1']

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.xavier_normal_(m.weight, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, input1, input2):
        out1 = self.forward_resnet(input1)
        # print(out1.shape)
        out2 = self.forward_emonet(input2)
        # print(out2.shape)
        if self.feat_fusion == 'concat':
            feat = torch.cat((out1, out2), 1)
        elif self.feat_fusion == "multiply":
            # Element wise multiplication
            feat = out1 * out2
        else:
            raise ValueError
        out = self.fc(feat)

        return {'out': out, 'pred': out[:, :-1], 'valence': out[:, -1], 'feat': feat}


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=12288, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        y = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1] * x.shape[3] * x.shape[4]))
        self.lstm.flatten_parameters()
        feat = self.lstm(y)[0]
        out = self.linear(feat)
        out = self.softmax(out)
        a1 = out.unsqueeze(-1)
        a2 = a1.unsqueeze(-1)
        a3 = torch.tile(a2, (x.shape[1], x.shape[3], x.shape[4]))
        out = torch.permute(a3, (0, 2, 1, 3, 4))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((5, 64, 64))
        self.max_pool = nn.AdaptiveMaxPool3d((5, 64, 64))
        self.ratio = ratio
        self.in_channels = in_channels

        self.fc = nn.Sequential(nn.Conv3d(self.in_channels, self.in_channels // self.ratio, 1, padding='same', bias=False),
                                nn.ReLU(),
                                nn.Conv3d(self.in_channels // self.ratio, self.in_channels, 1, padding='same', bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        # print("avg pool:", avg_out.shape)
        max_out = self.fc(self.max_pool(x))
        # print(max_out.shape)
        out = avg_out + max_out
        return self.sigmoid(out)


class MoodDeltaEmonet(nn.Module):
    """
        Encoder for mood: ResNet
        Encoder for emotion = EmoFAN
    """

    def __init__(self, feat_fusion, dropout_rate, is_pretrained, num_neurons_fc, num_mood_classes, num_delta_classes, cfg):
        super(MoodDeltaEmonet, self).__init__()

        # Fusion technique
        self.feat_fusion = feat_fusion

        self.dropout_rate = dropout_rate

        self.num_neurons_fc = num_neurons_fc

        self.num_mood_classes = num_mood_classes

        self.is_pretrained = is_pretrained

        # self.sp_att = SpatialAttention(kernel_size=3)
        # self.ch_att = ChannelAttention(in_channels=3, ratio=1)
        # self.temp_att = TemporalAttention(hidden_size=128, num_layers=2)

        self.resnet1 = generate_model_branchedlinear(model_depth=cfg["RESNET_DEPTH"]).to(cfg["DEVICE"])

        self.resnet2 = generate_model_branchedlinear(model_depth=cfg["RESNET_DEPTH"]).to(cfg["DEVICE"])

        # print(self.resnet)

        self.emofan = EmoNet(num_modules=2, n_expression=cfg["EMOFAN_EMOTIONS"], n_reg=2, n_blocks=4, attention=True,
                             temporal_smoothing=False).to(cfg["DEVICE"])
        # print(self.emofan)

        if self.is_pretrained:
            # Load weights to emonet
            state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{cfg["EMOFAN_EMOTIONS"]}.pth')
            print(f'Loading EmoFAN from {state_dict_path}')
            state_dict = torch.load(str(state_dict_path))
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.emofan.load_state_dict(state_dict, strict=False)

        # model.emo_fc_2[3] = nn.Linear(in_features=128, out_features=n_reg, bias=True)

        self.fc_in_features = self.emofan.emo_fc_2[0].in_features  # [bs, 256]

        # get features before the last linear block
        # Refer emonet supplementary material for more info
        self.emofan.emo_fc_2 = nn.Identity()    # Returns [bs, 256]

        # either consider mood and emotion features or mood, delta and emotion features
        assert self.feat_fusion in ['concat', 'multiply']

        if self.feat_fusion == "concat":
            feature_coeff = 3
        elif self.feat_fusion == "multiply":
            feature_coeff = 1
        else:
            raise ValueError

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * feature_coeff, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, num_mood_classes + num_delta_classes + num_neurons_fc, bias=False)
        )
        self.fc.apply(self.init_weights)

    def forward_emonet(self, x):
        output = self.emofan(x)
        output = output['feature'].view(output['feature'].size()[0], -1)  # [bs, 256]
        return output

    def forward_resnet1(self, x):
        output = self.resnet1(x)
        return output['out1']

    def forward_resnet2(self, x):
        output = self.resnet2(x)
        return output['out1']

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.xavier_normal_(m.weight, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, input1, input2):
        # out = self.ch_att(input1) * input1
        # out = self.temp_att(out) * out
        out1 = self.forward_resnet1(input1)
        out2 = self.forward_resnet2(input1)
        # print(out1.shape)
        out3 = self.forward_emonet(input2)
        # print(out2.shape)
        if self.feat_fusion == 'concat':
            feat = torch.cat((out1, out2, out3), 1)
        elif self.feat_fusion == "multiply":
            # Element wise multiplication
            feat = out1 * out2 * out3
        else:
            raise ValueError
        out = self.fc(feat)

        return {'out': out, 'mood': out[:, :-4], 'delta': out[:, 3:-1], 'valence': out[:, -1], 'feat': feat}


class TSNet(nn.Module):
    """
        Encoder for mood: ResNet
        Encoder for emotion = EmoFAN
    """

    def __init__(self, dropout_rate, is_pretrained, num_neurons_fc, cfg):
        super(TSNet, self).__init__()

        self.dropout_rate = dropout_rate

        self.num_neurons_fc = num_neurons_fc

        self.is_pretrained = is_pretrained

        self.emofan = EmoNet(num_modules=2, n_expression=cfg["EMOFAN_EMOTIONS"], n_reg=2, n_blocks=4, attention=True,
                             temporal_smoothing=False).to(cfg["DEVICE"])
        # print(self.emofan)

        if self.is_pretrained:
            # Load weights to emonet
            state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{cfg["EMOFAN_EMOTIONS"]}.pth')
            print(f'Loading EmoFAN from {state_dict_path}')
            state_dict = torch.load(str(state_dict_path))
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.emofan.load_state_dict(state_dict, strict=False)

        # model.emo_fc_2[3] = nn.Linear(in_features=128, out_features=n_reg, bias=True)

        self.fc_in_features = self.emofan.emo_fc_2[0].in_features  # [bs, 256]

        # get features before the last linear block
        # Refer emonet supplementary material for more info
        self.emofan.emo_fc_2 = nn.Identity()    # Returns [bs, 256]

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            #nn.Linear(128, 128, bias=True),
            #nn.BatchNorm1d(128),
            #nn.ReLU(inplace=True)

        )
        self.fc1.apply(self.init_weights)

        self.fc2 = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, num_neurons_fc, bias=False)
        )
        self.fc2.apply(self.init_weights)

    def forward_emonet(self, x):
        output = self.emofan(x)
        output = output['feature'].view(output['feature'].size()[0], -1)  # [bs, 256]
        return output

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.xavier_normal_(m.weight, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, input1):
        out1 = self.forward_emonet(input1)
        out2 = self.fc1(out1)
        # print(out2.shape)
        out3 = self.fc2(out2)

        return {'valence': out3, 'feat': out2}


if __name__ == "__main__":
    torch.manual_seed(5)
    n_reg = 1
    device = 'cuda:0'
    cfg = load_config(str(Path(__file__).parent.joinpath('config.yaml')))

    model = ResnetEmonet(feat_fusion='concat', dropout_rate=0.5, is_pretrained=True, num_neurons_fc=1,
                            num_mood_classes=3, cfg=cfg)
    model.to(device)
    # print(model)

    input_var1 = torch.randn(10, 3, 5, 64, 64).to(device)
    input_var2 = torch.randn(10, 3, 256, 256).to(device)

    out_dict = model(input_var1, input_var2)
    # print(out_dict['out'])
    print(out_dict['out'].shape)

    '''
    print("Parameters to learn")
    parameters_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            parameters_to_update.append(param)
            print("\t", name)
    '''
