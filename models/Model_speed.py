import torch.nn as nn
import math
import torch
from models.transformer import Transformer
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class PosEncoding(torch.nn.Module):
    def __init__(self,max_seq_len,d_model=512):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000,2.0 * (j//2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)]
        )
        pos_enc[:,0::2] = np.sin(pos_enc[:,0::2])
        pos_enc[:,1::2] = np.cos(pos_enc[:,1::2])
        pos_enc = pos_enc.astype(np.float32)
        self.pos_enc = torch.nn.Embedding(max_seq_len,d_model)
        self.pos_enc.weight = torch.nn.Parameter(torch.from_numpy(pos_enc),requires_grad=False)

    def forward(self, input_len):
        '''

        :param input_len: [7,7,7,7,...,7] shape=[batch_size]
        :return:
        '''
        input_pos = torch.tensor([list(range(0,len)) for len in input_len]).cuda()
        return self.pos_enc(input_pos)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Permutation(nn.Module):
    def __init__(self,d_model,classes):
        super(Permutation, self).__init__()
        self.d_model = d_model
        self.classes = classes

        self.classifier = nn.Sequential(*[nn.Linear(7*512,2048),nn.BatchNorm1d(2048),nn.ReLU(inplace=True),
                                          nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),
                                          nn.Linear(512,self.classes)])

        self.weights_init(self.classifier)

    def forward(self, input):
        output = self.classifier(input)
        return output

    def weights_init(self,model):
        if type(model) in [nn.ConvTranspose2d, nn.Linear]:
            nn.init.xavier_normal(model.weight.data)
            nn.init.constant(model.bias.data, 0.1)


class EST(nn.Module):
    def __init__(self, block, layers,clips=7,img_num_per_clip=5,d_model=512,nhead=4,dropout=0.1,encoder_nums=3,decoder_nums=3, use_norm=True,per_classes=10):
        self.inplanes = 64
        self.key_clips = 1
        self.per_classes = per_classes
        self.d_model = d_model
        self.clips = clips
        self.img_num_per_clip = img_num_per_clip
        self.use_norm = use_norm
        super(EST, self).__init__()
        ### First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(0.1)
        self.cos = nn.CosineSimilarity(dim=2)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)

        ### Second layer
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=encoder_nums,
                                                num_decoder_layers=decoder_nums
                                                , dropout=dropout)
        self.query_embed = torch.nn.Embedding(self.key_clips, self.d_model)
        self.pos_enc = PosEncoding(max_seq_len=self.clips, d_model=self.d_model)

        #############Third layer
        self.per_branch = Permutation(self.d_model,self.per_classes)

        ### Forth layer
        self.classifier = nn.Sequential(*[nn.Linear(512, 128), nn.ReLU(True),
                                          nn.Linear(128, 64), nn.ReLU(True),
                                          nn.Dropout(0.4),
                                          nn.Sequential(nn.Linear(64, 7))])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,per=torch.tensor([0,1,2,3,4,5,6])):

        video_feature = []
        b = x.size(0)
        x = x.view(b*35,3,224,224)
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        f = f.squeeze()
        #print(f.size())
        #f = f.view(self.img_num_per_clip,b*self.clips,self.d_model)
        #x = f.view(b,self.clips,self.img_num_per_clip,self.d_model)
        x = f.view(self.img_num_per_clip,b*self.clips,self.d_model)
        output,weight = self.attn(x,x,x)
        output = x + self.dropout3(output)
        output = self.norm1(output).permute(1,2,0)
        global_feature = self.maxpool2(output).view(1,self.clips*b,self.d_model)
        local_feature = output.permute(2,0,1)
        #dis_list = []
        #local_feature = local_feature.view(b*self.img_num_per_clip*self.clips,self.d_model)
        #print(local_feature.size(),global_feature.size())
        dis_alpha = self.cos(local_feature,global_feature).unsqueeze(dim=2)
        #print(dis_alpha.size())
        dis_alpha = dis_alpha.permute(1,2,0)
        #for i in range(self.img_num_per_clip):
        #    dis = self.cos(local_feature[i],global_feature)
        #    dis_list.append(dis)
        #dis_alpha = torch.stack(dis_list,dim=1).unsqueeze(dim=1)
        #print(dis_alpha.size())
        dis_alpha = torch.clamp(dis_alpha, min=1e-8)
        output = output.mul(dis_alpha).sum(2).div(dis_alpha.sum(2))
        #print(output.size())
        output.view(b,7,self.d_model)
        video_feature.append(output)
        #for i in range(self.clips):

        #    vs = []

        #    ff = x[:, :, :, :, :, i]  # x[batch_size,3,224,224, 5, 7]
        #    for j in range(self.img_num_per_clip):
        #        f = ff[:, :, :, :, j]  # ff[batch_size,3,224,224, 3]
                                 # ff[batch_size,3,224,224]

        #        f = self.conv1(f)
        #        f = self.bn1(f)
        #        f = self.relu(f)
        #        f = self.maxpool(f)

        #        f = self.layer1(f)
        #        f = self.layer2(f)
        #        f = self.layer3(f)
        #        f = self.layer4(f)
        #        f = self.avgpool(f)

        #        f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]

        #        vs.append(f)

        #    vs_stack = torch.stack(vs,dim=2)
        #    vs_stack = vs_stack.permute(2,0,1)
        #    output,weight = self.attn(vs_stack,vs_stack,vs_stack)
        #    if self.use_norm:
        #        output = vs_stack + self.dropout3(output)
        #        output = self.norm1(output).permute(1,2,0)

        #    global_feature = self.maxpool2(output).squeeze(dim=2)
        #    local_feature = output.permute(2,0,1) # 5*b*512
        #    dis_list = []
        #    for t in range(self.img_num_per_clip):
        #        dis = self.cos(local_feature[i],global_feature)
        #        dis_list.append(dis)
        #    dis_alpha = torch.stack(dis_list,dim=1).unsqueeze(dim=1)
        #    dis_alpha = torch.clamp(dis_alpha, min=1e-8)
        #    output = output.mul(dis_alpha).sum(2).div(dis_alpha.sum(2))

        #    video_feature.append(output)


        ori_video_feature = torch.stack(video_feature,dim=1)  #video_feature = bacth_size * clip_num * clip_feature的维度  即 bacth_size * 7 * 512        b = video_feature.size(0)
        b = int(ori_video_feature.size(0))
        pos = torch.tensor([self.clips] * b)
        pos_enc = self.pos_enc(pos)
        for i in range(b):
            pos_enc[i] = pos_enc[i][per[i]]
        x = ori_video_feature + pos_enc
        x = x.permute(1, 0, 2)
        tgt = self.query_embed.weight
        tgt = tgt.unsqueeze(1).repeat(1,b,1)
        emotion_clip_feature = self.transformer(x, tgt).permute(1, 2, 0)

        emotion_clip_feature = emotion_clip_feature.reshape(-1,self.d_model * self.key_clips)


        #####permutation
        #ori_video_feature_detach = ori_video_feature.detach()
        #per_feature = ori_video_feature_detach + emotion_clip_feature.unsqueeze(dim=1)
        #per_output = self.per_branch(per_feature.view(b,-1))

        output = self.classifier(emotion_clip_feature)

        return output


def resnet18_EST(pretrained=False, **kwargs):
    # Constructs base a ResNet-18 model.
    model = EST(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model