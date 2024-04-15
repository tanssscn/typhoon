import numpy as np
from torch.cuda.amp import autocast
from torch import nn
import sys
from tqdm import tqdm
from torch.cuda import amp

from typh_Generation.utils.datasetTest import TrainDataSetTest
import os
import argparse

import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import arg_config
from models.factory import create_model

from pathlib import Path
import utils.tools as util

from typh_Generation.utils.utils import increment_path


# num_layers也就是depth
def model_forward_multi_layer(model, inputs, targets_len=arg_config.targets_len, num_layers=2):
    states_down = [None] * num_layers
    states_up = [None] * num_layers

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)

    for i in range(targets_len):
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    global step
    model.train()
    loss_function = nn.MSELoss()
    # loss_function = torch.nn.MSELoss()
    accu_loss = []
    # 梯度清零，防止梯度累加
    optimizer.zero_grad()
    scaler = amp.GradScaler()

    # tqdm：进度条
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        inputs, targets = data
        with autocast():
            outputs = model_forward_multi_layer(model, inputs.to(device))
            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            img_len = outputs.shape[0]
            img_batch = outputs.shape[1]
            for i in range(img_len):
                for j in range(img_batch):
                    ii = outputs[i, j, :].clone().cpu()
                    ii = ii.permute(1, 2, 0) * 255.
                    util.save_image(ii.detach().numpy(), f'{arg_config.root}{i}{j}.png')
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1).to(device)
            loss = loss_function(outputs, targets_)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        accu_loss.append(loss.item())
        # loss.backward()
        # loss.detach()：返回一个新的tensor，它是loss的一个副本，并且不需要计算梯度。
        # accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}".format(epoch, np.mean(accu_loss))
        # 判断loss是否为无穷大（梯度爆炸），无穷大则需要退出训练
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        # optimizer.step()：更新模型参数。
        # optimizer.step()
        # optimizer.zero_grad()：清空梯度。
        optimizer.zero_grad()

    return np.mean(accu_loss)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.MSELoss()
    # loss_function = torch.nn.MSELoss()
    model.eval()
    accu_loss = []  # 累计损失

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        inputs, targets = data
        outputs = model_forward_multi_layer(model, inputs.to(device))
        outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
        # img_len = outputs.shape[0]
        # img_batch = outputs.shape[1]
        # for i in range(img_len):
        #     for j in range(img_batch):
        #         ii = outputs[i, j, :].clone().cpu()
        #         ii = ii.permute(1, 2, 0) * 255.
        #         util.save_image(ii.numpy(), f'{arg_config.root}{i}{j}.png')

        targets_ = torch.cat((inputs[:, 1:], targets), dim=1).to(device)
        loss = loss_function(outputs, targets_)
        accu_loss.append(loss.item())

        data_loader.desc = "[valid epoch {}] loss: {:.5f}".format(epoch, np.mean(accu_loss))

    return np.mean(accu_loss)


def main(args):
    # 使用GPU还是CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")  # 创建保存模型权重的文件夹：如果不存在weights文件夹，则创建该文件夹
    # 增量保存：args.project（runs/train）、args.name（exp）
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=False)  # increment run
    # SummaryWriter允许训练程序异步调用直接从训练循环向文件添加数据的方法，而不会减慢训练速度。
    tb_writer = SummaryWriter(save_dir)
    # 返回args.data_path路径下所有图片的路径和标签

    # 实例化训练数据集
    my_dataset = TrainDataSetTest(images_path=args.data_path,
                                  status="train")
    # 实例化验证数据集
    # val_dataset = TrainDataSet(images_path=os.path.join(args.data_path, 'val'),
    #                            status="val")

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 实例化训练数据加载器和验证数据加载器，并设置参数：batch_size、shuffle、pin_memory、num_workers、collate_fn
    # shuffle = True：打乱数据集,pin_memory = True：将数据从GPU缓存到CPU，以提高数据读取速度，
    # num_workers = 4：设置多进程加载数据，collate_fn = train_dataset.collate_fn：设置数据集的拼接方法，
    print(my_dataset)
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw,
    #                                            collate_fn=train_dataset.collate_fn)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_dataset.collate_fn)

    model_name = args.model
    weights_path = args.weights
    status = "train"
    # 创建模型，并加载预训练权重，如果有预训练权重，则加载预训练权重
    model = create_model(model_name, device, weights_path, status)
    # pg: 所有超参数
    pg = [p for p in model.parameters() if p.requires_grad]
    # 使用用AdamW优化器，学习率为0.0001，权重衰减为5E-2
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # 设置best_loss值为9999，当loss值小于该值时，将模型保存为model-best.pth文件
    best_loss = torch.as_tensor(9999)
    # 对每个折叠进行训练和验证
    kfold = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(my_dataset)):
        print(f"Fold {fold + 1}")
        # 根据索引划分训练集和验证集
        train_dataset = torch.utils.data.Subset(my_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(my_dataset, val_idx)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_dataset.collate_fn)
        # 开始训练
        for epoch in range(args.epochs):
            # train
            train_loss = train_one_epoch(model=model,
                                         optimizer=optimizer,
                                         data_loader=train_loader,
                                         device=device,
                                         epoch=epoch)

            # validate
            val_loss = evaluate(model=model,
                                data_loader=val_loader,
                                device=device,
                                epoch=epoch)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            # 将loss值写入tensorboard文件中，并保存模型权重文件
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            # 如果val_loss小于best_loss，将模型保存为model-best.pth文件
            if val_loss * 0.9 < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_dir + "/model-best.pth")


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--name', default='exp')
parser.add_argument('--model', default=arg_config.model)
parser.add_argument('--project', default=arg_config.project, help='save to project/name')
# 数据集所在根目录
parser.add_argument('--data-path', type=str, default=arg_config.data_path)
# 预训练权重路径，如果不想载入就设置为空
parser.add_argument('--weights', type=str, default=arg_config.pre_weights, help='initial weights path')
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args(args=[])

main(opt)
