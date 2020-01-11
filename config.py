CONFIG = {
    'data_path': "G:/无源感知研究/数据采集/2019_07_18/实验室(3t3r)(resample)(归一化)1.mat",
    'num_workers': 4,
    'pin_memory': True,
    'batch_size': 32,
    'epochs': 75,
    'lr': 1e-3,
    'momentum': .9,
    'l2_decay': 0.001,
    'lambda': 10,
    'sequence_max_len': 677,
    'input_feature': 270,
    'hidden_size': 300,
    'n_class': 6,
    'model_type':'lstm',
    'model_save_path': 'model_conv.pkl',
    'tensorboard_log_path': 'tensorboard_log/lab_lstm_h300_RMSprop_L2_1'
}
