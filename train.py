from ray import tune
import torch.optim as optim
from network.model import MyModel
from network.loss import MyLoss
from dataset.HighD_Dataset_DGL import HighD_Dataset
from dgl.dataloading import GraphDataLoader
from utils.util import AverageMeter

def train(model, optimizer, train_loader):
    losses = AverageMeter()
    for i, (X, Y, mask) in enumerate(train_loader):
        # 先不采用batch训练
        X_graph = X["graph"][0,...]
        X_feature = X["feature"][0,...]
        Y_graph = Y["graph"][0,...]
        Y_feature = Y["feature"][0,...]        
        mask = mask[0,...]


        output = model(X_graph, X_feature*mask)
        loss = MyLoss(output,Y_graph,Y_feature,mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #TODO: 梯度裁剪等操作        
        losses.update(loss.item())

    return losses.avg

def training_function(config):
    
    HighD_dataset = HighD_Dataset(X_len=20,X_step=1,Y_len=20,Y_step=1,diff=0,name='data_22',raw_dir='./')
    HighD_dataloader = GraphDataLoader(HighD_dataset, batch_size=1, shuffle=False)
    print("Dataset Ready!")

    model = MyModel(num_feats=8, output_dim=8, hidden_size=64, num_layers=3,seq_len=6, horizon=1, num_heads=8)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    #TODO: 学习率计划、早停等
    for epoch in range(100):
        # Iterative training function - can be any arbitrary training procedure.
        loss = train(model, optimizer, HighD_dataloader)
        # Feed the score back back to Tune.
        tune.track.log(mean_loss=loss)
        #TODO：checkpoint

# TODO：GPU训练
# analysis = tune.run(
#     training_function,
#     config={
#         "lr": tune.grid_search([0.001, 0.01, 0.1])
#     })

# print("Best config: ", analysis.get_best_config(
#     metric="mean_loss", mode="min"))

# # Get a dataframe for analyzing trial results.
# df = analysis.results_df

config=dict()
config['lr']=0.1
training_function(config)