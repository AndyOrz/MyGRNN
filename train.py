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

# from model.Model import MyModel
from network.model import MyModel
from network.loss import MyLoss
from dataset.HighD_Dataset_DGL_1graph import HighD_Dataset
from dgl.dataloading import GraphDataLoader
from utils.util import AverageMeter


class Trainer:
    def __init__(self, config, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = MyModel(num_feats=4, output_dim=4, hidden_size=64, num_layers=2,seq_len=10, horizon=1, device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.criterion = MyLoss(num_feats=4)
        #学习率计划√
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        epochs = config['epochs']
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # 先用了个适用于image classification的lr函数
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        self.scheduler.last_epoch = 0
        self.epoch = 0
        self.best_loss = 99999

        HighD_dataset = HighD_Dataset(X_len=10,X_step=1,Y_len=1,Y_step=1,diff=9,name='data_01',raw_dir='./dataset/')

        n_val = int(len(HighD_dataset) * config['val_percent'])
        n_train = len(HighD_dataset) - n_val
        train_dataset, val_dataset = random_split(HighD_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(2021))

        self.train_dataloader = GraphDataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=(self.device == "cuda"))
        self.val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=(self.device == "cuda"))
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

    def train(self):
        losses = AverageMeter()

        self.model.train()
        with tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (graph, X, Y, _mask) in enumerate(self.train_dataloader):
                # 先不采用batch训练
                X = X[0,...]
                Y = Y[0,...]
                # mask = mask[0,...]
                if X.shape[1]==0:
                    continue

                ###### 临时！ 测试用mask
                mask = torch.ones_like(X,dtype=torch.uint8)  #[10,N,2]
                mask[9,np.random.randint(mask.shape[1]),:]=0
                mask = mask.to(self.device)

                output = self.model(graph, X*mask)  #[1,N,2]
                loss = self.criterion(output,Y,mask)

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

    def eval(self):
        losses = AverageMeter()
        self.model.eval()

        with tqdm(total=len(self.val_dataloader),desc='Validation round') as pbar:
            for i, (graph, X, Y, _mask) in enumerate(self.val_dataloader):
                X = X[0,...]
                Y = Y[0,...]
                if X.shape[1]==0:
                    continue

                ###### 临时！ 测试用mask
                mask = torch.ones_like(X,dtype=torch.uint8)  #[10,N,2]
                mask[9,np.random.randint(mask.shape[1]),:]=0
                mask = mask.to(self.device)

                output = self.model(graph, X*mask)  #[1,N,2]
                loss = self.criterion(output,Y,mask)

                losses.update(loss.item())
                pbar.update()

        return losses.avg

# def training_function(config):

#     HighD_dataset = HighD_Dataset(X_len=10,X_step=1,Y_len=1,Y_step=1,diff=9,name='data_01',raw_dir='./dataset/')

#     n_val = int(len(HighD_dataset) * config['val_percent'])
#     n_train = len(HighD_dataset) - n_val
#     train_dataset, val_dataset = random_split(HighD_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(2021))

#     train_dataloader = GraphDataLoader(train_dataset, batch_size=1, shuffle=True)
#     val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=False)
#     print("Dataset Ready!")

#     device = config['device']
#     model = MyModel(num_feats=4, output_dim=4, hidden_size=64, num_layers=2,seq_len=10, horizon=1, device=device)
#     optimizer = optim.Adam(model.parameters(), lr=config["lr"])
#     criterion = MyLoss(num_feats=4)

#     best_loss = 99999
#     dir_checkpoint = config['dir_checkpoint']
#     epochs = config['epochs']
    

#     #TODO: 学习率计划√ 、早停等
#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # 先用了个适用于image classification的lr函数
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#     scheduler.last_epoch = 0

#     for epoch in range(epochs):
#         print("epoch:{}".format(epoch+1))
#         scheduler.step()
#         loss = train(model, criterion, optimizer, train_dataloader, device)
#         val_loss = eval(model, criterion, val_dataloader, device)
#         print('train loss = {}, validation loss = {}'.format(loss,val_loss))
#         # Feed the score back back to Tune.
#         # tune.track.log(mean_loss=loss)
#         #TODO：checkpoint √
#         try:
#             os.mkdir(dir_checkpoint)
#             print('Created checkpoint directory')
#         except OSError:
#             pass

#         if loss < best_loss or epoch + 1 == epochs :
#             torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}_loss_{str(loss)}.pth')
#             best_loss = loss
#             print(f'Checkpoint {epoch + 1} saved ! loss (train) = ' + str(loss))

# # TODO：多GPU训练
# # analysis = tune.run(
# #     training_function,
# #     config={
# #         "lr": tune.grid_search([0.001, 0.01, 0.1])
# #     })

# # print("Best config: ", analysis.get_best_config(
# #     metric="mean_loss", mode="min"))

# # # Get a dataframe for analyzing trial results.
# # df = analysis.results_df

# start_time=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# config=dict()
# # config['lr']=1e-3
# config['device']='cpu'
# config['dir_checkpoint']='ckpts/{}_{}/'.format(start_time,config['device'])
# config['epochs']=20
# # config['val_percent']=0.2
# training_function(config)

MAX_EPOCH = 20
CHECKPOINT_DIR = "ckpts/0403/"

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
        ctl = Trainer(modelconfig, device='cpu')

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
            train_loss = ctl.train()
            i += 1
            # 计时并计入
            info = "%d train_loss: %f" % (ctl.epoch, train_loss)
            print(info)
            logging.info(info)

            # 自动存档
            if i % args.fckp == 0:
                ctl.save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "ckp%d.pth" % ctl.epoch)
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
                val_loss = ctl.eval()
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