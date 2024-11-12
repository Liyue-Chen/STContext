

import torch
import argparse
import sys
UCTBfile="/home/pku/ltf/Spatialcontext/UCTB-master"
sys.path.insert(0,UCTBfile)
import os
from UCTB.utils.utils_MTGNN import load_dataset
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.dataset.dataloader2 import NodeTrafficLoader
# from UCTB.evaluation import metric
from UCTB.utils.utils_MTGNN import *
from UCTB.model.MTGNN_s_t import gtnet
import pdb
import time
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=50,help='step_size')


parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10,help='number of runs')

# data parameters
parser.add_argument("--dataset", default='Bike', type=str, help="configuration file path")
parser.add_argument("--city", default='DC', type=str)
parser.add_argument("--closeness_len", default=6, type=int)
parser.add_argument("--period_len", default=7, type=int)
parser.add_argument("--trend_len", default=4, type=int)
parser.add_argument("--data_range", default="all", type=str)
parser.add_argument("--train_data_length", default="all", type=str)
parser.add_argument("--test_ratio", default=0.2, type=float)
parser.add_argument("--MergeIndex", default=1, type=int)
parser.add_argument("--MergeWay", default="sum", type=str)
parser.add_argument("--Year", default=1, type=int)
parser.add_argument("--_City_type", default="none", type=str)
parser.add_argument("--T_method", default="one", type=str)
parser.add_argument("--S_method", default="mlp", type=str)
parser.add_argument("--k", default=1, type=int)
parser.add_argument("--mlplayer", default=2, type=int)
parser.add_argument("--value0", default=1, type=int)
parser.add_argument("--value1", default=200, type=int)
parser.add_argument("--distance_thereshhold", default=6000, type=int)
parser.add_argument("--hiddendim", default=35, type=int)
parser.add_argument("--gcnlayer", default=2, type=int)
parser.add_argument("--contextType", default='poi', type=str)
parser.add_argument("--poiooutdim", default=10, type=int)
parser.add_argument("--poitype", default=0, type=int)
parser.add_argument("--ALL1", default='False', type=str)
parser.add_argument("--theta", default=0, type=float)
parser.add_argument("--traintype", default='test', type=str)
parser.add_argument("--thres", default=1, type=int)
parser.add_argument("--interaction", default='none', type=str)
parser.add_argument("--data_mode", default='none', type=str)
parser.add_argument("--time_vector", default=1, type=int)



args = parser.parse_args()
args._City_type=args.dataset+args.city

# loading data
uctb_data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,data_dir='data3',
                                     data_range=args.data_range, train_data_length=args.train_data_length,
                                     test_ratio=float(args.test_ratio),
                                     closeness_len=args.closeness_len,
                                     period_len=args.period_len,
                                     trend_len=args.trend_len,
                                     normalize=True,
                                     MergeIndex=args.MergeIndex,
                                     poi_year=args.Year,
                                     City_type=args._City_type,
                                     MergeWay=args.MergeWay,
                                     value0=args.value0,
                                     value1=args.value1,
                                     poitype=args.poitype,
                                     ALL1=args.ALL1,
                                     theta=args.theta,
                                     contexttype=args.contextType,
                                     IIF_=args.S_method,
                                     data_mode=args.data_mode,
                                     )







# pdb.set_trace()
args.num_nodes = uctb_data_loader.station_number
args.seq_in_len = uctb_data_loader.closeness_len + uctb_data_loader.period_len + uctb_data_loader.trend_len
args.seq_out_len = 1

args.save = os.path.abspath('./Exp_1/ExperimentMerge{}_{}_{}/context_{}experiment_S_{}T_{}_k_{}_POI_outdim_{}_All1{}_theta_{}_{}/'.format(args.MergeIndex,args.dataset, args.city,args.contextType, args.S_method,args.T_method,args.k,args.poiooutdim,args.ALL1,args.theta,args.interaction))

if not os.path.exists(args.save):
    os.makedirs(args.save)


# Build Graph
graph_obj = GraphGenerator(graph='distance', data_loader=uctb_data_loader,threshold_distance=args.distance_thereshhold)


device = torch.device(args.device)
data_dict = load_dataset(uctb_data_loader, args.batch_size, args.batch_size, args.batch_size)
time_vec=args.time_vector


model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                  layer_norm_affline=False,graph_AM=graph_obj.AM[0],IIFdata=uctb_data_loader.IIF,
                  S_method=args.S_method ,T_method=args.T_method,poidim=uctb_data_loader.poi_dim,
                  poioutdim=args.poiooutdim,gcnlayer=args.gcnlayer,hiddendim=args.hiddendim,timevec=time_vec,interaction=args.interaction)


#Freeze model


def freeze_model(model, to_freeze_dict, keep_step=None):
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass

    return model



path = '/home/pku/ltf/Spatialcontext/UCTB-master/Experiments/MTGNN/NO_POI_60min/{}_experiment/MTGNN_{}_{}_{}_{}/exp_best_.pth'.format(args.dataset,args.dataset,args.city,args.MergeIndex,args.k)
save_model = torch.load(path)
model_dict = model.state_dict()




state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)
model.load_state_dict(model_dict)
model = freeze_model(model=model, to_freeze_dict=save_model)

#Freeze model

model = model.to(device)



print(args)
print('The recpetive field size is', model.receptive_field)
nParams = sum([p.nelement() for p in model.parameters()])
print('Number of model parameters is', nParams)
engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, device, args.cl)


print("start training...",flush=True)
his_loss =[]
val_time = []
train_time = []
minl = 1e5


for i in range(1,args.epochs+1):
    train_loss = []
    train_mape = []
    train_rmse = []
    t1 = time.time()
    data_dict['train_loader'].shuffle()
    for iter, (x, y) in enumerate(data_dict['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainx= trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        # pdb.set_trace()
        if iter%args.step_size2==0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes/args.num_split)
        for j in range(args.num_split):
            if j != args.num_split-1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = trainx[:, :, id, :]
            ty = trainy[:, :, id, :]
            metrics = engine.train(tx, ty[:,0,:,:],args.S_method,args.T_method,id,args.traintype)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
        if iter % args.print_every == 0 :
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
    t2 = time.time()
    train_time.append(t2-t1)
    #validation
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x, y) in enumerate(data_dict['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        metrics = engine.eval(testx, testy[:,0,:,:])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    s2 = time.time()
    log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    val_time.append(s2-s1)
    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
    if mvalid_loss<minl:
        torch.save(engine.model.state_dict(), args.save + "/exp" + str(args.expid) + '_' + str(i) +'_' + ".pth")
        minl = mvalid_loss



print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
bestid = np.argmin(his_loss)

print('bestID',bestid)
engine.model.load_state_dict(torch.load(args.save + "/exp" + str(args.expid) + '_' + str(bestid+1) +'_' + ".pth"))
print("Training finished")
print("The valid loss on best model is", str(round(his_loss[bestid],4)))



#test data


outputs = []
realy = torch.Tensor(data_dict['y_test']).to(device)
realy = realy.transpose(1, 3)[:, 0, :, :]
for iter, (x, y) in enumerate(data_dict['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    with torch.no_grad():
        engine.model.eval()
        preds = engine.model(testx)
        preds = preds.transpose(1, 3)
    outputs.append(preds.squeeze())
y_truth = uctb_data_loader.normalizer.inverse_transform(uctb_data_loader.test_y)
yhat = torch.cat(outputs, dim=0).cpu().numpy()
yhat = yhat[:realy.size(0), ...]
yhat = uctb_data_loader.normalizer.inverse_transform(yhat)
yhat = yhat[...,np.newaxis]






ones_indices = np.where(uctb_data_loader.Index_Divide==1)[0]
twos_indices = np.where(uctb_data_loader.Index_Divide==2)[0]
threes_indices = np.where(uctb_data_loader.Index_Divide == 3)[0]



from UCTB.evaluation.metric import rmse,mape,mae,trunc_rmse,trunc_smape



loss1_rmse_test=rmse(target=y_truth[:,ones_indices,:],prediction=yhat[:,ones_indices,:])
print('scene1_rmse_test:',loss1_rmse_test)

loss2_rmse_test=rmse(target=y_truth[:,twos_indices,:],prediction=yhat[:,twos_indices,:])
print('scene2_rmse_test:',loss2_rmse_test)




loss1_mape_test=trunc_smape(target=y_truth[:,ones_indices,:],prediction=yhat[:,ones_indices,:],threshold=args.thres)
print('scene1_smape_test:',loss1_mape_test)

loss2_mape_test=trunc_smape(target=y_truth[:,twos_indices,:],prediction=yhat[:,twos_indices,:],threshold=args.thres)
print('scene2_smape_test:',loss2_mape_test)


print('Test RMSE',rmse(yhat,y_truth))
print('Test MAE',mae(yhat,y_truth,1))
print('Test SMAPE',trunc_smape(yhat,y_truth,args.thres))
np.save('MTGNN_Bike_Chicago_test_pred',yhat)
np.save('MTGNN_Bike_Chicago_test_truth',y_truth)





def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已成功创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")


folder_name = './result/{}_{}_{}/context_{}_S_{}T_{}_k_{}_POI_outdim_{}_{}_{}'.format(args.MergeIndex,args.dataset, args.city,args.contextType, args.S_method,args.T_method,args.k,args.poiooutdim,args.interaction,args.data_mode)
create_folder_if_not_exists(folder_name)


np.save(folder_name+'/index.pkl',uctb_data_loader.Index_Divide)
np.save(folder_name+'/pre.pkl',yhat)
np.save(folder_name+'/truth.pkl',y_truth)












