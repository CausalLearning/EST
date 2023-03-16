from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
import load_materials
from models.Model import resnet18_EST
from tensorboardX import SummaryWriter
import pytorch_warmup as warmup
import random
from options.base_options import BaseOptions
from torch.backends import cudnn
import os


def train(train_loader, model, criterion, optimizer, epoch, opt, writer):
    running_loss, count, correct_count, running_cls_loss, running_per_loss, correct_per_count = 0., 0, 0., 0., 0., 0.
    model.train()
    for i, data in enumerate(train_loader):
        target_first = data['label']
        input_var = torch.autograd.Variable(data['data_shuffle'])
        order_label = data['per_label']
        target = target_first.cuda(non_blocking=True)
        order_label = order_label.cuda(non_blocking=True)

        target_var = torch.autograd.Variable(target)
        order_label = torch.autograd.Variable(order_label)
        pred_score, per_score = model(input_var, per=data['per_shuffle'])

        # compute gradient and do Adam step
        loss_cls = criterion(pred_score, target_var)
        loss_per = criterion(per_score, order_label)
        loss = loss_cls + loss_per * opt.lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss
        running_loss += loss.item()
        running_cls_loss += loss_cls.item()
        running_per_loss += loss_per.item()
        correct_count += (torch.max(pred_score, dim=1)[1] == target_var).sum()
        correct_per_count += (torch.max(per_score, dim=1)[1] == order_label).sum()
        count += input_var.size(0)

        if i % opt.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t Cls_Acc{acc:.4f}\t Per_Acc{per_acc:.4f}\t Loss cls {loss_cls:.4f}\t Loss per {loss_per:.4f}'
                    .format(epoch, i, len(train_loader), loss=running_loss / count, acc=int(correct_count) / count,
                            per_acc=int(correct_per_count) / count, loss_cls=running_cls_loss / count,
                            loss_per=running_per_loss / count))
    print(
        ' Train_Acc {train_Video:.4f}\t  Train_Loss {Train_Loss:.4f}\t Per_Acc{per_acc:.4f}\t Loss cls {loss_cls:.4f}\t Loss per {loss_per:.4f}'.
            format(train_Video=int(correct_count) / count, Train_Loss=running_loss / count,
                   per_acc=int(correct_per_count) / count, loss_cls=running_cls_loss / count,
                   loss_per=running_per_loss / count))

    writer.add_scalar('final_loss', running_loss / count, epoch)
    writer.add_scalar('final_cls_loss', running_cls_loss / count, epoch)
    writer.add_scalar('final_cls_acc', int(correct_count) / count, epoch)
    writer.add_scalar('final_per_loss', running_per_loss / count, epoch)
    writer.add_scalar('final_per_acc', int(correct_per_count) / count, epoch)


def validate(val_loader, model,args):
    model.eval()
    test_correct_count, test_count, test_correct_per_count, test_per_acc = 0, 0, 0, 0.

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            ########################test shuffle
            input_var = torch.autograd.Variable(data['data_shuffle'])
            order_label = data['per_label']
            order_label = order_label.cuda(non_blocking=True)
            order_label = torch.autograd.Variable(order_label)

            # compute output
            _, per_score = model(input_var, per=data['per_shuffle'])
            ####################################################

            #################### test cls
            input_var = torch.autograd.Variable(data['data_normal'])
            target_first = data['label']
            # compute output
            target = target_first.cuda(non_blocking=True)

            target_var = torch.autograd.Variable(target)

            pred_score, _ = model(input_var, per=data['per_normal'])
            #####################################################
            #if torch.max(pred_score, dim=1)[1] != target_var:
            #    print(data['path'], '  ', utils.cate2label(args.dataset_name)[torch.max(pred_score, dim=1)[1].item()])
            test_correct_count += (torch.max(pred_score, dim=1)[1] == target_var).sum()
            test_correct_per_count += (torch.max(per_score, dim=1)[1] == order_label).sum()

            test_count += input_var.size(0)

            if args.draw_weight:
                video_path_list = data['path'][0].split('/')
                video_path = video_path_list[-2]+'/'+video_path_list[-1]
                heat_map_path = os.path.join(args.heat_map_path,args.name,video_path)
                trans_path = os.path.join(heat_map_path,'trans')
                cos_path = os.path.join(heat_map_path,'cos')
                utils.mkdirs(trans_path)
                utils.mkdirs(cos_path)
                for t in range(len(weight_list)):
                    heat = os.path.join(trans_path,str(t)+'.npy')
                    utils.draw_weight(weight_list[t].squeeze().detach().cpu().numpy(),heat)
                for t in range(len(cos_weight)):
                    heat = os.path.join(cos_path,str(t)+'.npy')
                    utils.draw_weight(cos_weight[t].detach().cpu().numpy(),heat)

        test_acc = int(test_correct_count) / test_count
        test_per_acc = int(test_correct_per_count) / test_count
        print(' Test_Acc: {test_Video:.4f} '.format(test_Video=test_acc))
        print(' Test_per_Acc: {test_per_Video:.4f} '.format(test_per_Video=test_per_acc))

        return test_acc, test_per_acc


def main(opt):
    train_loader, val_loader = load_materials.LoadDataset(opt)
    model = resnet18_EST(clips=opt.snippets, img_num_per_clip=opt.per_snippets, d_model=opt.d_model, nhead=opt.nhead,
                         encoder_nums=opt.encoder_nums, decoder_nums=opt.decoder_nums, use_norm=opt.use_norm,
                         per_classes=opt.permutation_classes,draw_weight=opt.draw_weight)
    if opt.isTrain and not opt.continue_train:
        model = load_materials.LoadParameter(model, opt.parameterDir)
        print('train !')
    elif opt.continue_train:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(opt.pre_train_model_path)['state_dict'])
        print('load eval model !')
    else:
        print('load eval model !')
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(opt.eval_model_path)['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if not opt.isTrain:
        validate(val_loader, model,opt)
        return

    per_branch_params = list(map(id, model.module.per_branch.parameters()))
    base_params = filter(lambda p: id(p) not in per_branch_params and p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.module.per_branch.parameters(), 'lr': opt.lr}
    ], lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)

    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_count)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=opt.warm_up)
    warmup_scheduler.last_step = -1
    best_prec1 = 0.
    for epoch in range(opt.epoch, opt.epochs_count):
        lr_schduler.step(epoch)
        warmup_scheduler.dampen()
        train(train_loader, model, criterion, optimizer, epoch, opt, writer)
        prec1, per_acc = validate(val_loader, model,opt)

        writer.add_scalar('final_test_acc', prec1, epoch)
        writer.add_scalar('final_test_per_acc', per_acc, epoch)
        is_best = prec1 > best_prec1
        if is_best:
            print('better model!')
            best_prec1 = max(prec1, best_prec1)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': prec1,
            }, opt)
        else:
            print('Model too bad & not save')


if __name__ == '__main__':
    opt = BaseOptions().parse()

    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    torch.manual_seed(opt.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(opt.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(opt.seed)  # 为所有GPU设置随机种子
    random.seed(opt.seed)

    writer = SummaryWriter(comment=opt.name)

    main(opt)

    writer.close()
