
import os

#不同建模技术
folder_name = './UbiComptxtnew/0605_60min_BikeNYC_'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method mlp --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method gcn --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method att2_cross --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method IIF --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method gcn --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method att2_cross --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --S_method IIF --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))


    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --S_method mlp --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --S_method gcn --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --S_method att2_cross --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --S_method IIF --T_method none --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))


# #不同context在不同数据集的实验



for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType road --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType population --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType district --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))


for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType population --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType road --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType district --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))


for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --interaction concat --thres 5 --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.3 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --interaction concat --thres 5 --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType population --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.3 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --interaction concat --thres 5 --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType road --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.3 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --interaction concat --thres 5 --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType district --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.3 '.format(l))



for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType population --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType road --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType district --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))




for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType population --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType road --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --MergeWay average --thres 1 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType district --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.09 '.format(l))




# Temporal only
for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 20 --S_method none --interaction none --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 3 --S_method none --interaction none --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 3 --S_method none --interaction none --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 10 --S_method none --interaction none --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 10 --S_method none --interaction none --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))


#End_to_end train


for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype train_base --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype train_base --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype train_base --batch_size 32 --poiooutdim 10 --theta 0.3 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype train_base --batch_size 32 --poiooutdim 10 --theta 0.1 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype train_base --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))


#Pretrain &fine-tune train


for l in range(0,5):
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Metro --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Bike --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.25 '.format(l))
    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset Taxi --city NYC --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.3 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 12 --dataset PEMS --city BAY --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.1 '.format(l))

    os.system('python MTGNN_Scontext_Tcontext.py --thres 5 --interaction concat --S_method mlp --T_method time-concat --ALL1 False --value0 1 --value1 200 --contextType poi --MergeIndex 1 --dataset Pedestrian --city Melbourne --k {} --gcnlayer 2 --hiddendim 20 --traintype test --batch_size 32 --poiooutdim 10 --theta 0.2 '.format(l))
