import argparse
from UCTB.dataset import NodeTrafficLoader
from UCTB.dataset.context_loader import STContextLoader
import os
from UCTB.utils.utils_MTGNN_st import load_dataset,load_dataset_forecast
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.dataset import NodeTrafficLoader
# from UCTB.evaluation import metric
from UCTB.utils.utils_MTGNN_st import *
from UCTB.model.MTGNN_st_context_forecast import gtnet
import pdb
import time
from UCTB.evaluation.metric import rmse,mape,mae

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

# context modeling parameters:
parser.add_argument('--context_hidden_size',type=int,default=32)
parser.add_argument('--temporal_modeling_method',type=str,default='mlp')
parser.add_argument('--spatial_modeling_method',type=str,default='no')



parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=10,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10,help='number of runs')
parser.add_argument('--mark',type=str,default='',help='')
parser.add_argument('--pretrain_epochs',type=int,default=300,help='number of runs')
parser.add_argument('--train_mode',type=str,default='end2end',help='number of runs')



# data parameters
parser.add_argument("--dataset", default='METR', type=str, help="configuration file path")
parser.add_argument("--city", default='LA', type=str)
parser.add_argument("--closeness_len", default=6, type=int)
parser.add_argument("--period_len", default=7, type=int)
parser.add_argument("--trend_len", default=4, type=int)
parser.add_argument("--data_range", default="all", type=str)
parser.add_argument("--train_data_length", default="all", type=str)
parser.add_argument("--test_ratio", default=0.2, type=float)
parser.add_argument("--MergeIndex", default=12, type=int)
parser.add_argument("--MergeWay", default="sum", type=str)
parser.add_argument("--remove", type=str_to_bool,default=True)
# context data_loader arguments
parser.add_argument('--context_historical_window',default=1,type=int)
parser.add_argument('--context_future_window',default=0,type=int)
parser.add_argument('--is_multicolumn',type=str_to_bool,default=False)



args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic =False
torch.backends.cudnn.benchmark =False
# loading data
# loading crowd flow data
uctb_data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,data_dir='dataset',
                                     data_range=args.data_range, train_data_length=args.train_data_length,
                                     test_ratio=float(args.test_ratio),
                                     closeness_len=args.closeness_len,
                                     period_len=args.period_len,
                                     trend_len=args.trend_len,
                                     normalize=True,
                                     MergeIndex=args.MergeIndex,
                                     MergeWay=args.MergeWay,remove=args.remove)
# loading context data

stcontextloader = STContextLoader()
if 'PEMS' in args.dataset:
    name = 'bay_speed'
if 'Pedestrian' in args.dataset:
    name = 'mel_pedestrian'
if 'Metro' in args.dataset:
    name = 'nyc_metro'
if ('Bike' in args.dataset) or 'Taxi' in args.dataset:
    name = 'nyc_bikeandtaxi'
context_args = {'name':name,'type':'Forecast Weather','is_multicolumn':args.is_multicolumn}

stcontextloader.get_stcontext(args=context_args)

expected_station_info = []

lat_lng_list = np.array([[float(e1) for e1 in e[2:4]]
                                     for e in uctb_data_loader.dataset.node_station_info])

new_latlng_array = lat_lng_list[uctb_data_loader.traffic_data_index]
for s in range(new_latlng_array.shape[0]):
    expected_station_info.append((new_latlng_array[s,0],new_latlng_array[s,1]))

# pdb.set_trace()
expected_time_range = uctb_data_loader.dataset.time_range
expected_time_fitness = uctb_data_loader.dataset.time_fitness
stcontextloader.TTransformation(expected_time_fitness=expected_time_fitness,expected_time_range=expected_time_range)
stcontextloader.TTransformation_forecast(expected_time_fitness=expected_time_fitness,expected_time_range=expected_time_range)
print('transformation finished')
assignment_matrix,assignment_matrix_forecast = stcontextloader.get_assignment_matrix(expected_station_info)
stcontextloader.st_context_transformed = np.transpose(np.array(list(stcontextloader.latlon2context.values())),(1,0,2))
stcontextloader.st_context_forecast_transformed = np.transpose(np.array(list(stcontextloader.latlon2context_current.values())),(1,0,2))

stcontextloader.get_feature(args.context_historical_window,uctb_data_loader.train_index,uctb_data_loader.test_index,future_window=args.context_future_window)
stcontextloader.train_st_context[np.isnan(stcontextloader.train_st_context)] = 0
stcontextloader.test_st_context[np.isnan(stcontextloader.test_st_context)] = 0
# pdb.set_trace()
stcontextloader.train_st_context_forecast[np.isnan(stcontextloader.train_st_context_forecast)] = 0
stcontextloader.test_st_context_forecast[np.isnan(stcontextloader.test_st_context_forecast)] = 0
# pdb.set_trace()
# parameters resetting


args.num_nodes = uctb_data_loader.station_number
args.seq_in_len = uctb_data_loader.closeness_len + uctb_data_loader.period_len + uctb_data_loader.trend_len
args.seq_out_len = 1
args.save = os.path.abspath('./experiment/MTGNN_{}_{}_{}_{}/{}/'.format(args.dataset, args.city, args.MergeIndex,args.seed,'{}_{}'.format('si' if args.spatial_modeling_method=='no' else 'sv', 'p' if args.context_future_window==-1 else 'c')))
context_window_size = args.context_historical_window # 1 for current time slot
context_window_size_forecast = args.context_future_window+1
# pdb.set_trace()
context_dimension = stcontextloader.train_st_context.shape[-1]
context_dimension_forecast = stcontextloader.train_st_context_forecast.shape[-1]
print('Context Dimensions:',context_dimension)
print('Context Dimensions Forecast:',context_dimension_forecast)
if not os.path.exists(args.save):
    os.makedirs(args.save)
    
# Build Graph
# graph_obj = GraphGenerator(graph='distance', data_loader=uctb_data_loader)


device = torch.device(args.device)
data_dict = load_dataset_forecast(uctb_data_loader, args.batch_size, args.batch_size, args.batch_size,st_context_loader=stcontextloader)
# 需要改下
# predefined_A = graph_obj.AM[0]
# predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
# predefined_A = predefined_A.to(device)

model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, context_window_size=context_window_size,context_dimensions=context_dimension,context_hidden_size=args.context_hidden_size,temporal_modeling_method=args.temporal_modeling_method,spatial_modeling_method=args.spatial_modeling_method,context_stations=assignment_matrix.shape[0],context_stations_forecast=assignment_matrix_forecast.shape[0],context_dimensions_forecast=context_dimension_forecast,context_window_size_forecast=context_window_size_forecast)
# pdb.set_trace()
save_model = torch.load('experiment/MTGNN_{}_{}_{}_base_{}/best.pth'.format(args.dataset,args.city,args.MergeIndex,args.seed))
def freeze_model(model, to_freeze_dict, keep_step=None):
    with torch.no_grad():
        for (name, param) in model.named_parameters():
            # pdb.set_trace()

            if name in to_freeze_dict:
                param.data.copy_(to_freeze_dict[name])
                # print(name,to_freeze_dict[name])
                if 'end_conv' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # for (name, param) in model.named_parameters():
        #     print(name,torch.mean(param))
    return model

# dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model = freeze_model(model=model, to_freeze_dict=save_model)

model = model.to(device)
print(args)
print('The recpetive field size is', model.receptive_field)
nParams = sum([p.nelement() for p in model.parameters()])
print('Number of model parameters is', nParams)
engine = Trainer_context_forecast(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, device, False)
outputs = []
realy = torch.Tensor(data_dict['y_test']).to(device)
realy = realy.transpose(1, 3)[:, 0, :, :]
am = torch.Tensor(assignment_matrix).to(device)
amf = torch.Tensor(assignment_matrix_forecast).to(device)
for iter, (x,context,context_forecast, y) in enumerate(data_dict['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    testc = torch.Tensor(context).to(device)
    testc = testc.transpose(1, 3)
    engine.model.eval()
    engine.model.temporal_modeling.eval()
    engine.model.channel_modeling_1.eval()
    engine.model.gate_linear.eval()
    engine.model.end_conv_1.eval()
    engine.model.end_conv_2.eval()
    
    with torch.no_grad():

        preds = engine.model(testx)
        preds = preds.transpose(1, 3)

    outputs.append(preds.squeeze())
y_truth = uctb_data_loader.normalizer.inverse_transform(uctb_data_loader.test_y)
yhat = torch.cat(outputs, dim=0).cpu().numpy()
yhat = yhat[:realy.size(0), ...]
yhat = uctb_data_loader.normalizer.inverse_transform(yhat)
yhat = yhat[...,np.newaxis]
print('Model Performance Before finetune')
print('Test RMSE',rmse(yhat,y_truth,0))
print('Test MAE',mae(yhat,y_truth,0))
print('Test MAPE',mape(yhat,y_truth,1))
# pdb.set_trace()
print("start training...",flush=True)
his_vrmse =[]
val_time = []
train_time = []
minl = 1e5
# pdb.set_trace()
# engine.model.zero_grad()
# if args.train_mode != 'end2end':
#     engine.model.enter_mode('pretrain')
engine.model.enter_mode('together')
historical_i = 0
his_trmse = []
for i in range(1,args.epochs+1):
    train_loss = []
    train_mape = []
    train_rmse = []
    t1 = time.time()
    data_dict['train_loader'].shuffle()
    # if i == args.pretrain_epochs:
    #     engine.model.load_state_dict(torch.load(args.save + "/exp" + str(args.expid) + '_' + str(historical_i) +'_' + ".pth"))
    #     engine.model.enter_mode('finetune')
    for iter, (x,context,context_forecast, y) in enumerate(data_dict['train_loader'].get_iterator()):
        # pdb.set_trace()
        trainx = torch.Tensor(x).to(device)
        trainx= trainx.transpose(1, 3)
        trainc = torch.Tensor(context).to(device)
        trainc = trainc.transpose(1, 3)
        traincf = torch.Tensor(context_forecast).to(device)
        traincf = traincf.transpose(1, 3)
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
                tc = trainc
                tcf = traincf
                ty = trainy[:, :, id, :]
                # if i < args.pretrain_epochs:
                #     metrics = engine.train(tx, ty[:,0,:,:],id)
                # else:
                # pdb.set_trace()
                metrics = engine.train((tx,tc,tcf,am,amf), ty[:,0,:,:],id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
        # if iter % args.print_every == 0 :
            # log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            # print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
    t2 = time.time()
    train_time.append(t2-t1)
    #validation
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x,context,context_forecast, y) in enumerate(data_dict['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testc = torch.Tensor(context).to(device)
        testc = testc.transpose(1, 3)
        testcf = torch.Tensor(context_forecast).to(device)
        testcf = testcf.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        if i <= args.pretrain_epochs:
            metrics = engine.eval(testx, testy[:,0,:,:])
        else:
            metrics = engine.eval((testx,testc,testcf,am,amf), testy[:,0,:,:])
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
    his_vrmse.append(mvalid_rmse)
    his_trmse.append(mtrain_rmse)
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
    if mvalid_rmse<minl and i>1:
        historical_i = i
        print('update')
        torch.save(engine.model.state_dict(), args.save + "/best.pth")
        minl = mvalid_rmse
import pickle as pkl
with open('historical_train_loss_{}.pkl'.format(args.train_mode),'wb') as fp:
    pkl.dump(his_trmse,fp)
with open('historical_val_loss_{}.pkl'.format(args.train_mode),'wb') as fp:
    pkl.dump(his_vrmse,fp)
print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
bestid = np.argmin(his_vrmse)
engine.model.load_state_dict(torch.load(args.save + "/best.pth"))
print("Training finished")
print("The valid loss on best model is", str(round(his_vrmse[bestid],4)))
with torch.no_grad():
    for name, param in engine.model.named_parameters(): 
        print(name,torch.mean(param))
# #valid data
# outputs = []
# realy = torch.Tensor(data_dict['y_val']).to(device)
# realy = realy.transpose(1,3)[:,0,:,:]
# for iter, (x, y) in enumerate(data_dict['val_loader'].get_iterator()):
#     testx = torch.Tensor(x).to(device)
#     testx = testx.transpose(1,3)
#     with torch.no_grad():
#         preds = engine.model(testx)
#         preds = preds.transpose(1,3)
#     outputs.append(preds.squeeze())
# yhat = torch.cat(outputs,dim=0)
# yhat = yhat[:realy.size(0),...]
# pred = yhat
# vmae, vmape, vrmse = metric(pred,realy)
#test data
outputs = []
realy = torch.Tensor(data_dict['y_test']).to(device)
realy = realy.transpose(1, 3)[:, 0, :, :]
for iter, (x,context,context_forecast, y) in enumerate(data_dict['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    testc = torch.Tensor(context).to(device)
    testc = testc.transpose(1, 3)
    testcf = torch.Tensor(context_forecast).to(device)
    testcf = testcf.transpose(1, 3)
    engine.model.eval()
    engine.model.temporal_modeling_forecast.eval()
    engine.model.temporal_modeling.eval()
    engine.model.channel_modeling_1.eval()
    engine.model.channel_modeling_1_forecast.eval()
    engine.model.gate_linear.eval()
    engine.model.gate_linear_forecast.eval()
    engine.model.end_conv_1.eval()
    engine.model.end_conv_2.eval()
    if engine.model.spatial_modeling_method != 'no':
        engine.model.spatial_modeling.eval()
        engine.model.spatial_modeling_forecast.eval()
    if engine.model.spatial_modeling_method != 'no':
        engine.model.spatial_modeling.eval()
        engine.model.spatial_modeling_forecast.eval()
    with torch.no_grad():
        if args.epochs <= args.pretrain_epochs:
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        else:
            preds = engine.model(testx,testc,testcf,am,amf)
            preds = preds.transpose(1, 3)
    outputs.append(preds.squeeze())
y_truth = uctb_data_loader.normalizer.inverse_transform(uctb_data_loader.test_y)
yhat = torch.cat(outputs, dim=0).cpu().numpy()
yhat = yhat[:realy.size(0), ...]
yhat = uctb_data_loader.normalizer.inverse_transform(yhat)
yhat = yhat[...,np.newaxis]
print('{} {}'.format(args.dataset,args.city))
print('Test RMSE',rmse(yhat,y_truth,0))
print('Test MAE',mae(yhat,y_truth,0))
print('Test MAPE',mape(yhat,y_truth,1))
if not os.path.exists('Result/'+str(args.seed)):
    os.mkdir('Result/'+str(args.seed))
prefix = 'Result/'+str(args.seed)+'/'
if args.context_future_window == -1 and args.spatial_modeling_method != 'mlps':
    np.save(prefix+'MTGNN_{}_{}_sinvariant_past_test_pred_{}.npy'.format(args.dataset,args.city,args.mark),yhat)
    np.save(prefix+'MTGNN_{}_{}_sinvariant_past_test_truth_{}.npy'.format(args.dataset,args.city,args.mark),y_truth)
elif args.context_future_window == 0 and args.spatial_modeling_method != 'mlps':
    np.save(prefix+'MTGNN_{}_{}_sinvariant_current_test_pred_{}.npy'.format(args.dataset,args.city,args.mark),yhat)
    np.save(prefix+'MTGNN_{}_{}_sinvariant_current_test_truth_{}.npy'.format(args.dataset,args.city,args.mark),y_truth)
elif args.context_future_window == -1 and args.spatial_modeling_method == 'mlps':
    np.save(prefix+'MTGNN_{}_{}_svarying_past_test_pred_{}.npy'.format(args.dataset,args.city,args.mark),yhat)
    np.save(prefix+'MTGNN_{}_{}_svarying_past_test_truth_{}.npy'.format(args.dataset,args.city,args.mark),y_truth)
elif args.context_future_window == 0 and args.spatial_modeling_method == 'mlps':
    np.save(prefix+'MTGNN_{}_{}_svarying_current_test_pred_{}.npy'.format(args.dataset,args.city,args.mark),yhat)
    np.save(prefix+'MTGNN_{}_{}_svarying_current_test_truth_{}.npy'.format(args.dataset,args.city,args.mark),y_truth)
