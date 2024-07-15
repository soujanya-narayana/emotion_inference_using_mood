import torch.nn as nn
from pathlib import Path
import torch
from emonet import EmoNet
from delta_model import SiameseNetEmoNetMLP
from mood_model import generate_model_branchedlinear


class LinearNetEmoMoodDelta(nn.Module):
    """
        Encoder = EmoFAN
    """

    def __init__(self, is_pretrained_mood, is_pretrained_delta, is_pretrained_emofan, feat_fusion, num_neurons_fc, dropout_rate, cfg):
        super(LinearNetEmoMoodDelta, self).__init__()

        # Fusion technique
        self.feat_fusion = feat_fusion

        self.is_pretrained_mood = is_pretrained_mood
        self.is_pretrained_delta = is_pretrained_delta
        self.is_pretrained_emofan = is_pretrained_emofan

        self.dropout_rate = dropout_rate

        # either consider mood and emotion features or mood, delta and emotion features
        assert self.feat_fusion in ['mood_emotion', 'mood_delta_emotion']

        # This is a pretrained mood model
        self.mood_model = generate_model_branchedlinear(model_depth=cfg["RESNET_DEPTH"], n_classes_b1=cfg["CLASS_NUM_BRANCH1"],
                                                        n_classes_b2=cfg["CLASS_NUM_BRANCH2"],
                                                        dropout_rate=cfg["DROPOUT_RATE"]).to(cfg["DEVICE"])
        self.mood_model.eval()

        # This is Ravi's MTCLAR model
        self.delta_model = SiameseNetEmoNetMLP(emofan_n_emotions=cfg["EMOFAN_EMOTIONS"], is_pretrained_emofan=cfg["IS_PRETRAINED_EMOFAN"],
                                               feat_fusion=cfg["FEAT_FUSION_MTCLAR"], num_neurons=cfg["NUM_NEURONS_MTCLAR"],
                                               dropout_rate=cfg["DROPOUT_RATE"], is_multi_task=True).to(cfg["DEVICE"])

        self.delta_model.eval()

        # Pretrained EmoFAN model
        self.emofan = EmoNet(num_modules=2, n_expression=cfg["EMOFAN_EMOTIONS"], n_reg=2, n_blocks=4, attention=True,
                             temporal_smoothing=False).to(cfg["DEVICE"])
        self.emofan.eval()

        if self.is_pretrained_mood:
            # Load weights to mood model
            state_dict_path1 = Path(__file__).parent.joinpath('pretrained', f'mood_best_model.pth')
            print(f'Loading mood model from {state_dict_path1}...')
            state_dict1 = torch.load(str(state_dict_path1), map_location=torch.device('cuda:0'))
            state_dict1 = {k.replace('module.', ''): v for k, v in state_dict1.items()}
            self.mood_model.load_state_dict(state_dict1, strict=False)

            for param in self.mood_model.parameters():
                param.requires_grad = False

        if self.is_pretrained_delta:
            # Load weights to delta model
            state_dict_path2 = Path(__file__).parent.joinpath('pretrained', f'delta_best_model.pth')
            print(f'Loading delta model from {state_dict_path2}...')
            state_dict2 = torch.load(str(state_dict_path2), map_location=torch.device('cuda:0'))
            state_dict2 = {k.replace('module.', ''): v for k, v in state_dict2.items()}
            self.delta_model.load_state_dict(state_dict2, strict=False)

            for param in self.delta_model.parameters():
                param.requires_grad = False

        if self.is_pretrained_emofan:
            # Load weights to emofan
            state_dict_path3 = Path(__file__).parent.joinpath('pretrained', f'emonet_8.pth')
            print(f'Loading emofan model from {state_dict_path3}...')
            state_dict3 = torch.load(str(state_dict_path3), map_location=torch.device('cuda:0'))
            state_dict3 = {k.replace('module.', ''): v for k, v in state_dict3.items()}
            self.emofan.load_state_dict(state_dict3, strict=False)

            for param in self.emofan.parameters():
                param.requires_grad = False

        # model.emo_fc_2[3] = nn.Linear(in_features=128, out_features=n_reg, bias=True)

        self.fc_in_features = self.emofan.emo_fc_2[0].in_features  # [bs, 256]
        self.mood_features = self.mood_model.fc_b1_1[6]  # [bs, 256]
        self.delta_features = self.delta_model.fc0_1[11]  # [bs, 512]

        # get features before the last linear block
        # Refer emonet supplementary material for more info
        self.emofan.emo_fc_2 = nn.Identity()    # Returns [bs, 256]

        if self.feat_fusion == "mood_emotion":
            feature_coeff = 2
        elif self.feat_fusion == "mood_delta_emotion":
            feature_coeff = 4
        else:
            raise ValueError

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_in_features * feature_coeff, 1024, bias=True),
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
            nn.Linear(128, num_neurons_fc),
        )
        # initialize the weights
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # torch.nn.init.normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Linear):
            # torch.nn.init.normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def forward_mood(self, x):
        output = self.mood_model(x)
        output = output['feat1'].view(output['feat1'].size()[0], -1)
        # print("Mood output", output.shape)
        return output

    def forward_delta(self, x1, x2):
        output = self.delta_model(x1, x2)
        output = output['feat0'].view(output['feat0'].size()[0], -1)
        # print("Delta output", output.shape)
        return output

    def forward_emofan(self, x):
        output = self.emofan(x)
        output = output['feature'].view(output['feature'].size()[0], -1)  # [bs, 256]
        # print("Emofan output", output.shape)
        return output

    def forward(self, clip, image):
        # get two images' features
        with torch.no_grad():
            output1 = self.forward_mood(clip)  # (bs, 5, 3, 64, 64)
            output2 = self.forward_delta(clip[:, :, 0, :, :], clip[:, :, -1, :, :])
            output3 = self.forward_emofan(image)

        if self.feat_fusion == "mood_emotion":
            # concatenate both images' features
            output = torch.cat((output1, output3), 1)
            # print("Concatenated output", output.shape)
        elif self.feat_fusion == "mood_delta_emotion":
            # Take the absolute value between the two features
            output = torch.cat((output1, output2, output3), 1)
            # print("Concatenated output", output.shape)
        else:
            raise ValueError

        # pass the concatenation to the linear layers
        target = self.fc(output)

        return {'target': target}


class LinearNetVectors(nn.Module):
    """
        Encoder = EmoFAN
    """

    def __init__(self, feat_fusion, num_neurons_fc, dropout_rate):
        super(LinearNetVectors, self).__init__()

        # Fusion technique
        self.feat_fusion = feat_fusion

        self.dropout_rate = dropout_rate

        # either consider mood and emotion features or mood, delta and emotion features
        assert self.feat_fusion in ['mood_emotion', 'mood_delta_emotion']

        if self.feat_fusion == "mood_emotion":
            feature_coeff = 2
        elif self.feat_fusion == "mood_delta_emotion":
            feature_coeff = 4
        else:
            raise ValueError

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            # nn.Dropout(self.dropout_rate),
            nn.Linear(256 * feature_coeff, 256, bias=True),
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
            nn.Linear(128, num_neurons_fc),
        )
        # initialize the weights
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # torch.nn.init.normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Linear):
            # torch.nn.init.normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def forward(self, mood_vec, delta_vec, emofan_vec, previous_mood_vec):

        if self.feat_fusion == "mood_emotion":
            # concatenate both images' features
            print("Mood vec", mood_vec.dtype)
            mood_vec_updated = torch.add(mood_vec, previous_mood_vec)
            print("Mood vec updated", mood_vec_updated.dtype)
            output = torch.cat((mood_vec_updated, emofan_vec), 1)
            print("Concatenated output", output.dtype)
        elif self.feat_fusion == "mood_delta_emotion":
            print("Mood vec", mood_vec.dtype)
            # Take the absolute value between the two features
            mood_vec_updated = torch.add(mood_vec, previous_mood_vec)
            print("Mood vec updated", mood_vec_updated.dtype)
            output = torch.cat((mood_vec_updated, delta_vec, emofan_vec), 1)
            print("Concatenated output", output.dtype)
        else:
            raise ValueError

        # pass the concatenation to the linear layers
        target = self.fc(output)

        return {'target': target, 'output': output}


if __name__ == "__main__":
    torch.manual_seed(5)
    n_reg = 1
    device = 'cuda:0'
    model = LinearNetVectors(feat_fusion='mood_emotion', num_neurons_fc=1, dropout_rate=0.1)
    model.to(device)
    print(model)

    input_var1 = torch.rand(10, 256).to(device)
    input_var2 = torch.rand(10, 512).to(device)
    input_var3 = torch.rand(10, 256).to(device)
    input_var4 = torch.rand(10, 256).to(device)
    out_dict = model(input_var1, input_var2, input_var3, input_var4)
    print(out_dict['target'])
    print(out_dict['target'].shape)

    print("Parameters to learn")
    parameters_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            parameters_to_update.append(param)
            print("\t", name)
    """
    y = (torch.rand(10) < 0.5).int()
    y[y == 0] = -1
    print(y)
    lo = nn.CosineEmbeddingLoss(margin=0.5, reduction='none')(out_dict['output1'], out_dict['output2'], y.to(device))
    print(lo)
    """