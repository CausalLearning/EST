from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
from data.random_shuffle_dataset import RandomShuffleDataset
import utils


def LoadDataset(opt):
    cate2label = utils.cate2label(opt.dataset_name)

    train_dataset = RandomShuffleDataset(
        video_root=opt.train_video_root,
        video_list=opt.train_list_root,
        rectify_label=cate2label,
        isTrain= True,
        transform=transforms.Compose([transforms.ToTensor()]),
        opt=opt
    )

    val_dataset = RandomShuffleDataset(
        video_root=opt.test_video_root,
        video_list=opt.test_list_root,
        rectify_label=cate2label,
        isTrain = False,
        transform=transforms.Compose([transforms.ToTensor()]),
        opt=opt
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_threads,
         pin_memory=True, drop_last=True)   #True若数据集大小不能被batch_size整除，则删除最后一个不完整的批处理。

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size, shuffle=False,num_workers=opt.num_threads,
         pin_memory=True)

    return train_loader, val_loader


def LoadParameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (key == 'module.feature.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
