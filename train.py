import math
import os
from tqdm import tqdm
import argparse
import json
import logging
import traceback
from datetime import datetime

from ray import tune
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.utils.data import random_split

# from dataset.HighD_Dataset_DGL_1graph import HighD_Dataset
# from network.model_single_graph import MyModel
from dataset.HighD_Dataset_DGL import HighD_Dataset
from network.model_multi_graph import MyModel
from network.loss import MyLoss

from dgl.dataloading import GraphDataLoader
from utils.util import AverageMeter


class Trainer:
    def __init__(self, config, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = MyModel(num_feats=config['num_feats'], output_dim=config['num_feats'], hidden_size=64, num_layers=2,seq_len=10, horizon=1, device=self.device, bidirectional=True).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.criterion = MyLoss(num_feats=config['num_feats']).to(self.device)
        #学习率计划√
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        epochs = config['epochs']
        # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # 先用了个适用于image classification的lr函数
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        # self.scheduler.last_epoch = 0
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=0, last_epoch=-1)
        self.epoch = 0
        self.best_loss = 99999

        HighD_dataset = HighD_Dataset(X_len=10,X_step=1,Y_len=1,Y_step=1,diff=9,name='data_01',raw_dir='./dataset/', preprocess_all=True,device=self.device)

        n_val = int(len(HighD_dataset) * config['val_percent'])
        n_train = len(HighD_dataset) - n_val
        train_dataset, val_dataset = random_split(HighD_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(2021))

        self.train_dataloader = GraphDataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=(self.device == "cuda"))
        self.val_dataloader = GraphDataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=(self.device == "cuda"))
        print("Dataset Ready!")

    def save_checkpoint(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]

    def train_multi_graph(self):
        losses = AverageMeter()

        self.model.train()
        with tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (X, Y) in enumerate(self.train_dataloader):
                # graph batch train

                # enable mask
                X[-1].ndata['feature']=X[-1].ndata['feature']*X[-1].ndata['mask']
                # get output from model
                output = self.model(X,Y) 

                # select data and calculate loss
                predict = output[0,X[-1].ndata['mask']==0].view(-1,self.model._num_feats)
                truth = Y[0].ndata['feature'][X[-1].ndata['mask']==0].view(-1,self.model._num_feats)
                loss = self.criterion(predict,truth)
                loss = loss.sum()

                self.optimizer.zero_grad()
                loss.backward()
                #TODO: 梯度裁剪等操作 √
                # clip_grad_norm(model.parameters(), max_norm=10) 一个超参需要讨论
                self.optimizer.step()
                losses.update(loss.item())
                pbar.set_description('Loss: {:.2f}'.format(loss.item()))
                pbar.update()
        
        self.epoch += 1
        return losses.avg

    def eval_multi_graph(self):
        losses = AverageMeter()
        self.model.eval()

        with tqdm(total=len(self.val_dataloader),desc='Validation round') as pbar:
            for i, (X, Y) in enumerate(self.train_dataloader):
                # graph batch train

                # enable mask
                X[-1].ndata['feature']=X[-1].ndata['feature']*X[-1].ndata['mask']
                # get output from model
                output = self.model(X,Y) 

                # select data and calculate loss
                predict = output[0,X[-1].ndata['mask']==0].view(-1,self.model._num_feats)
                truth = Y[0].ndata['feature'][X[-1].ndata['mask']==0].view(-1,self.model._num_feats)
                loss = self.criterion(predict,truth)
                loss = loss.mean()

                losses.update(loss.item())
                pbar.update()

        return losses.avg



MAX_EPOCH = 20
CHECKPOINT_DIR = "ckpts/0405_1/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图卷积神经网络模型训练")

    parser.add_argument("--config", default="ModelConfig.json", help="要创建模型的参数文件")
    # parser.add_argument("--datafile", default="temp", help="数据集文件夹")
    parser.add_argument("--ckp", help="要载入的存档文件")
    parser.add_argument("--epoch", type=int, help="要训练到的epoch数")
    parser.add_argument("--fckp", type=int, default=1, help="存档频率")
    parser.add_argument("--fval", type=int, default=1, help="评估频率")
    args = parser.parse_args()

    # 日志文件
    fh = logging.FileHandler(filename="Train.log", encoding="utf-8")
    logging.basicConfig(level=logging.DEBUG, handlers=[fh])
    logging.info(
        "\n*********************** %s **************************\n" % datetime.now()
    )
    logging.info("调用参数：" + str(args))
    assert args.fckp > 0 and args.fval > 0, "评估或存档频率错误。"
    print("存档频率：%s，评估频率：%d" % (args.fckp, args.fval))
    print("Strat Time: %s" % datetime.now())

    try:
        if args.epoch is not None:
            MAX_EPOCH = args.epoch

        # 读取模型设置
        with open(args.config, "r") as f:
            modelconfig = json.load(f)
            modelconfig['epochs'] = MAX_EPOCH

        # 检查参数与数据集匹配性 ×

        # 创建训练器
        ctl = Trainer(modelconfig, device='cuda')

        # 读取存档
        if args.ckp is not None:
            ctl.load_checkpoint(args.ckp)

        # 确保存档目录有效
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            logging.debug("创建存档目录:%s" % CHECKPOINT_DIR)

        # 开始训练
        print("==========开始训练（%d/%d）==========" % (ctl.epoch, MAX_EPOCH))
        i = 0  # 仅用于自动存档、自动评估，不是epoch
        while ctl.epoch < MAX_EPOCH:
            train_loss = ctl.train_multi_graph()
            i += 1
            # 计时并计入
            info = "%d train_loss: %f" % (ctl.epoch, train_loss)
            print(info)
            logging.info(info)

            # 自动存档
            if i % args.fckp == 0:
                ctl.save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "ckpt{}_loss_{:.4f}.pth" .format(ctl.epoch,train_loss))
                )
                with open(os.path.join(CHECKPOINT_DIR, "ckp.json"), "w") as f:
                    json.dump(
                        modelconfig,
                        fp=f,
                        sort_keys=True,
                        indent=2,
                    )

                logging.info("checkpoint saved in %s." % CHECKPOINT_DIR)

            # 自动评估
            if i % args.fval == 0:
                val_loss = ctl.eval_multi_graph()
                print("val_loss: %f " % (val_loss))
                logging.info(
                    "%d val_loss: %f " % (ctl.epoch, val_loss)
                )

    except Exception as e:
        # 记录错误信息
        traceback.print_exc()
        logging.error(traceback.format_exc())

    finally:
        logging.info("End Time: %s" % datetime.now())
        print("End Time: %s" % datetime.now())