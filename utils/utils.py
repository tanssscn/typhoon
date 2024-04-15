import sys
import torch
from tqdm import tqdm
from pathlib import Path
import glob
import re


class DIST_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        device = targets.device  # 获取目标数据所在的设备
        loss = torch.zeros(1).to(device)  # 初始化损失为0
        for i in range(preds.shape[0]):  # 遍历预测数据的行数
            p = preds[i]  # 获取当前预测数据
            t = targets[i]  # 获取当前目标数据
            dist = (torch.pow((p[0] - t[0]), 2) + torch.pow((p[1] - t[1]), 2)).sqrt().unsqueeze(0).to(
                device)  # 计算预测数据与目标数据之间的距离
            loss = torch.cat((loss, dist), 0)  # 将距离追加到损失张量中
        loss = loss.mean()  # 计算平均损失
        return loss  # 返回损失值


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    global step
    model.train()
    loss_function = DIST_loss()
    # loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    # 梯度清零，防止梯度累加
    optimizer.zero_grad()

    sample_num = 0
    # tqdm：进度条
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        # images.shape[0]：获取batch_size
        sample_num += images.shape[0]

        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        # loss.detach()：返回一个新的tensor，它是loss的一个副本，并且不需要计算梯度。
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}".format(epoch, accu_loss.item() / (step + 1))
        # 判断loss是否为无穷大（梯度爆炸），无穷大则需要退出训练
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        # optimizer.step()：更新模型参数。
        optimizer.step()
        # optimizer.zero_grad()：清空梯度。
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = DIST_loss()
    # loss_function = torch.nn.MSELoss()
    model.eval()
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.5f}".format(epoch, accu_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1)
