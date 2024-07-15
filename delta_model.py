import torch
import torch.nn as nn
from emonet import EmoNet
from pathlib import Path


class SiameseNetEmoNetMLP(nn.Module):
    """
        Encoder = EmoFAN
    """

    def __init__(self, emofan_n_emotions, is_pretrained_emofan, feat_fusion, dropout_rate, num_neurons, is_multi_task=False):
        super(SiameseNetEmoNetMLP, self).__init__()

        # Fusion technique
        self.feat_fusion = feat_fusion

        self.is_pretrained_emofan = is_pretrained_emofan

        # concatenate features, take absolute difference between the features,
        # or do elementwise multiplication of the features.
        assert self.feat_fusion in ["concat", "absolute", "multiply"]

        self.is_multi_task = is_multi_task
        # Number of neurons in the last layer
        self.num_neurons = num_neurons

        self.emo_net = EmoNet(num_modules=2, n_expression=emofan_n_emotions, n_reg=2, n_blocks=4, attention=True,
                              temporal_smoothing=False)

        if self.is_pretrained_emofan:
            # Load weights to emonet
            state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{emofan_n_emotions}.pth')
            print(f'Loading EmoFAN from {state_dict_path}')
            state_dict = torch.load(str(state_dict_path))
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.emo_net.load_state_dict(state_dict, strict=False)

        # model.emo_fc_2[3] = nn.Linear(in_features=128, out_features=n_reg, bias=True)

        self.fc_in_features = self.emo_net.emo_fc_2[0].in_features  # [bs, 256]

        # get features before the last linear block
        # Refer emonet supplementary material for more info
        self.emo_net.emo_fc_2 = nn.Identity()    # Returns [bs, 256]

        if self.feat_fusion == "concat":
            feature_coeff = 2
        elif (self.feat_fusion == "absolute") or (self.feat_fusion == "multiply"):
            feature_coeff = 1
        else:
            raise ValueError

        self.dropout_rate = dropout_rate

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_in_features * feature_coeff, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, num_neurons[1]))

        # initialize the weights
        self.fc.apply(self.init_weights)

        if self.is_multi_task:
            # Branch for multitask learning
            self.fc0_1 = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.fc_in_features * feature_coeff, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2048, 1024, bias=True),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(1024, 512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
            )

            self.fc0_2 = nn.Sequential(
                nn.Linear(512, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128,
                          self.num_neurons[0]),
            )

            self.fc0_1.apply(self.init_weights)
            self.fc0_2.apply(self.init_weights)

            self.fc2 = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.fc_in_features * feature_coeff, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2048, 1024, bias=True),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(1024, 512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128,
                          self.num_neurons[0]),
            )
            self.fc2.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            #torch.nn.init.normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Linear):
            # torch.nn.init.normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.emo_net(x)
        output = output['feature'].view(output['feature'].size()[0], -1)  # [bs, 256]
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.feat_fusion == "concat":
            # concatenate both images' features
            output = torch.cat((output1, output2), 1)
        elif self.feat_fusion == "absolute":
            # Take the absolute value between the two features
            output = torch.abs(output1 - output2)
        elif self.feat_fusion == "multiply":
            # Element wise multiplication
            output = output1 * output2
        else:
            raise ValueError

        if self.is_multi_task:
            # pass the concatenation to the linear layers
            # Similarity Head
            target = self.fc(output)

            # Delta valence Head
            feat0 = self.fc0_1(output)
            target0 = self.fc0_2(feat0)

            # Delta Arousal Head
            target2 = self.fc2(output)

            return {'output1': output1, 'output2': output2, 'target': target, 'target0': target0, 'feat0': feat0,
                    'target2': target2}

        else:
            # pass the concatenation to the linear layers
            target = self.fc(output)

            return {'output1': output1, 'output2': output2, 'target': target}
