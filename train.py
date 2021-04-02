import math
import os
import time
from tqdm import tqdm

from ray import tune
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.utils.data import random_split

# from model.Model import MyModel
from network.model import MyModel
from network.loss import MyLoss
from dataset.HighD_Dataset_DGL_1graph import HighD_Dataset
from dgl.dataloading import GraphDataLoader
from utils.util import AverageMeter

def train(model, criterion ,optimizer, train_dataloader, device):
    losses = AverageMeter()

    model.to(device)
    model.train()
    with tqdm(total=len(train_dataloader)) as pbar:
        for i, (graph, X, Y, _mask) in enumerate(train_dataloader):
            # 先不采用batch训练
            graph = graph.to(device,non_blocking=True)
            X = X[0,...].to(device,non_blocking=True)
            Y = Y[0,...].to(device,non_blocking=True)
            # mask = mask[0,...].to(device)
            if X.shape[1]==0:
                continue

            ###### 临时！ 测试用mask
            mask = torch.ones_like(X,dtype=torch.uint8)  #[10,N,2]
            mask[9,np.random.randint(mask.shape[1]),:]=0
            mask = mask.to(device)

            output = model(graph, X*mask)  #[1,N,2]
            loss = criterion(output,Y,mask)

            optimizer.zero_grad()
            loss.backward()
            #TODO: 梯度裁剪等操作 √
            # clip_grad_norm(model.parameters(), max_norm=10) 一个超参需要讨论
            optimizer.step()
            losses.update(loss.item())
            pbar.set_description('Loss: {:.2f}'.format(loss.item()))
            pbar.update()

    return losses.avg

def eval(model, criterion, val_dataloader, device):
    losses = AverageMeter()
    model.to(device)
    model.eval()

    with tqdm(total=len(val_dataloader),desc='Validation round') as pbar:
        for i, (graph, X, Y, _mask) in enumerate(val_dataloader):
            # 先不采用batch训练
            graph = graph.to(device)
            X = X[0,...].to(device)
            Y = Y[0,...].to(device)
            # mask = mask[0,...].to(device)
            if X.shape[1]==0:
                continue

            ###### 临时！ 测试用mask
            mask = torch.ones_like(X,dtype=torch.uint8)  #[10,N,2]
            mask[9,np.random.randint(mask.shape[1]),:]=0
            mask = mask.to(device)

            output = model(graph, X*mask)  #[1,N,2]
            loss = criterion(output,Y,mask)

            losses.update(loss.item())
            pbar.update()

    return losses.avg

def training_function(config):

    HighD_dataset = HighD_Dataset(X_len=10,X_step=1,Y_len=1,Y_step=1,diff=9,name='data_01',raw_dir='./dataset/')

    n_val = int(len(HighD_dataset) * config['val_percent'])
    n_train = len(HighD_dataset) - n_val
    train_dataset, val_dataset = random_split(HighD_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(2021))

    train_dataloader = GraphDataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=False)
    print("Dataset Ready!")

    device = config['device']
    model = MyModel(num_feats=4, output_dim=4, hidden_size=64, num_layers=2,seq_len=10, horizon=1, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = MyLoss(num_feats=4)

    best_loss = 99999
    dir_checkpoint = config['dir_checkpoint']
    epochs = config['epochs']
    

    #TODO: 学习率计划√ 、早停等
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # 先用了个适用于image classification的lr函数
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = 0

    for epoch in range(epochs):
        print("epoch:{}".format(epoch+1))
        scheduler.step()
        loss = train(model, criterion, optimizer, train_dataloader, device)
        val_loss = eval(model, criterion, val_dataloader, device)
        print('train loss = {}, validation loss = {}'.format(loss,val_loss))
        # Feed the score back back to Tune.
        # tune.track.log(mean_loss=loss)
        #TODO：checkpoint √
        try:
            os.mkdir(dir_checkpoint)
            print('Created checkpoint directory')
        except OSError:
            pass

        if loss < best_loss or epoch + 1 == epochs :
            torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}_loss_{str(loss)}.pth')
            best_loss = loss
            print(f'Checkpoint {epoch + 1} saved ! loss (train) = ' + str(loss))

# TODO：多GPU训练
# analysis = tune.run(
#     training_function,
#     config={
#         "lr": tune.grid_search([0.001, 0.01, 0.1])
#     })

# print("Best config: ", analysis.get_best_config(
#     metric="mean_loss", mode="min"))

# # Get a dataframe for analyzing trial results.
# df = analysis.results_df

start_time=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
config=dict()
config['lr']=1e-3
config['device']='cpu'
config['dir_checkpoint']='ckpts/{}_{}/'.format(start_time,config['device'])
config['epochs']=20
config['val_percent']=0.2
training_function(config)