import os

# BikeNYC
os.system('python MTGNN_st.py --dataset Bike_2017-01-01-2017-07-01 --city NYC --MergeIndex 12 --MergeWay sum --device cuda:0 --batch_size 32 --epochs 300 --seed 1 --remove True')
os.system('python MTGNN_st_context.py --dataset Bike_2017-01-01-2017-07-01 --city NYC --batch_size 32 --context_historical_window 3 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method no --MergeIndex 12 --MergeWay sum --device cuda:0 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark shuffle --seed 1 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context.py --dataset Bike_2017-01-01-2017-07-01 --city NYC --batch_size 32 --context_historical_window 3 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method mlps --MergeIndex 12 --MergeWay sum --device cuda:0  --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark shuffle --seed 1 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context_forecast.py --dataset Bike_2017-01-01-2017-07-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_historical_window 3 --context_future_window 0 --spatial_modeling_method no --MergeIndex 12 --MergeWay sum --device cuda:0  --pretrain_epochs 1 --epochs 50 --train_mode pretrain --mark forecast --seed 1 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context_forecast.py --dataset Bike_2017-01-01-2017-07-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_historical_window 3 --context_future_window 0 --spatial_modeling_method mlps --MergeIndex 12 --MergeWay sum --device cuda:0  --pretrain_epochs 1 --epochs 50 --train_mode pretrain --mark forecast --seed 1 --is_multicolumn True --remove True')

# TaxiNYC
os.system('python MTGNN_st.py --dataset Taxi_2017-01-01-2017-07-01 --city NYC --MergeIndex 12 --MergeWay sum --device cuda:1 --batch_size 32 --epochs 500 --seed 1 --remove True')
os.system('python MTGNN_st_context.py --dataset Taxi_2017-01-01-2017-07-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method no --MergeIndex 12 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 700 --train_mode pretrain --mark shuffle --seed 1 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context.py --dataset Taxi_2017-01-01-2017-07-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method mlps --MergeIndex 12 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 700 --train_mode pretrain --mark shuffle --seed 1 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context_forecast.py --dataset Taxi_2017-01-01-2017-07-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_historical_window 1 --context_future_window 0 --spatial_modeling_method no --MergeIndex 12 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 700 --train_mode pretrain --mark forecast --seed 1 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context_forecast.py --dataset Taxi_2017-01-01-2017-07-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_historical_window 1 --context_future_window 0 --spatial_modeling_method mlps --MergeIndex 12 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 700 --train_mode pretrain --mark forecast --seed 1 --is_multicolumn True --remove True')

# PEMSBAY
os.system('python MTGNN_st.py --dataset PEMS_2017-01-01-2017-07-01 --city BAY --batch_size 32 --epochs 500 --MergeIndex 12 --MergeWay average --remove True --device cuda:1 --seed 3')
os.system('python MTGNN_st_context.py --dataset PEMS_2017-01-01-2017-07-01 --city BAY --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method no --MergeIndex 12 --MergeWay average --device cuda:1 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark shuffle --seed 1001 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context.py --dataset PEMS_2017-01-01-2017-07-01 --city BAY --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method mlps --MergeIndex 12 --MergeWay average --device cuda:1 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark shuffle --seed 1001 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context_forecast.py --dataset PEMS_2017-01-01-2017-07-01 --city BAY --batch_size 32 --temporal_modeling_method mlp --context_future_window 0 --spatial_modeling_method no --MergeIndex 12 --MergeWay average --device cuda:1 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark forecast --seed 1001 --is_multicolumn True --remove True')
os.system('python MTGNN_st_context_forecast.py --dataset PEMS_2017-01-01-2017-07-01 --city BAY --batch_size 32 --temporal_modeling_method mlp --context_future_window 0 --spatial_modeling_method mlps --MergeIndex 12 --MergeWay average --device cuda:1 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark forecast --seed 1001 --is_multicolumn True --remove True')

# Pedestrian_MEL
p_m_epochs = ' --epochs 1000'
os.system('python MTGNN_st_have_NaNs.py --dataset Pedestrian_2022-02-01-2022-08-01 --city Melbourne --batch_size 32 --epochs 300  --MergeIndex 1 --MergeWay sum --remove False --device cuda:1 --seed 401 --num_split 2')
os.system('python MTGNN_st_context_have_NaNs.py --dataset Pedestrian_2022-02-01-2022-08-01 --context_historical_window 1 --city Melbourne --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method no --MergeIndex 1 --MergeWay sum --remove False --device cuda:0 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark finetune --seed 401 --is_multicolumn False --remove False --num_split 2' + p_m_epochs)
os.system('python MTGNN_st_context_have_NaNs.py --dataset Pedestrian_2022-02-01-2022-08-01 --context_historical_window 1 --city Melbourne --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method mlps --MergeIndex 1 --MergeWay sum  --device cuda:0 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark finetune --seed 401 --is_multicolumn False --remove False --num_split 2' + p_m_epochs)
os.system('python MTGNN_st_context_forecast_have_NaNs.py --dataset Pedestrian_2022-02-01-2022-08-01 --context_historical_window 1 --city Melbourne --batch_size 32 --temporal_modeling_method mlp --context_future_window 0 --spatial_modeling_method no --MergeIndex 1 --MergeWay sum  --device cuda:0 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark finetune --seed 401 --is_multicolumn False --remove False --num_split 2' + p_m_epochs)
os.system('python MTGNN_st_context_forecast_have_NaNs.py --dataset Pedestrian_2022-02-01-2022-08-01 --context_historical_window 1 --city Melbourne --batch_size 32 --temporal_modeling_method mlp --context_future_window 0 --spatial_modeling_method mlps --MergeIndex 1 --MergeWay sum --device cuda:0 --pretrain_epochs 1 --epochs 300 --train_mode pretrain --mark finetune --seed 401 --is_multicolumn False --remove False --num_split 2' + p_m_epochs)
# MetroNYC
os.system('python MTGNN_st_have_NaNs.py --dataset Metro_2022-02-01-2022-08-01 --city NYC --batch_size 32 --epochs 400  --MergeIndex 1 --MergeWay sum --remove True --device cuda:1 --seed 1001')
os.system('python MTGNN_st_context_have_NaNs.py --dataset Metro_2022-02-01-2022-08-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method no --MergeIndex 1 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 500 --train_mode pretrain --mark shuffle --seed 1001 --is_multicolumn True --remove False')
os.system('python MTGNN_st_context_have_NaNs.py --dataset Metro_2022-02-01-2022-08-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_future_window -1 --spatial_modeling_method mlps --MergeIndex 1 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 500 --train_mode pretrain --mark shuffle --seed 1001 --is_multicolumn True --remove False')
os.system('python MTGNN_st_context_forecast_have_NaNs.py --dataset Metro_2022-02-01-2022-08-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_future_window 0 --spatial_modeling_method no --MergeIndex 1 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 500 --train_mode pretrain --mark forecast --seed 1001 --is_multicolumn True --remove False')
os.system('python MTGNN_st_context_forecast_have_NaNs.py --dataset Metro_2022-02-01-2022-08-01 --city NYC --batch_size 32 --temporal_modeling_method mlp --context_future_window 0 --spatial_modeling_method mlps --MergeIndex 1 --MergeWay sum --device cuda:1 --pretrain_epochs 1 --epochs 500 --train_mode pretrain --mark forecast --seed 1001 --is_multicolumn True --remove False')
