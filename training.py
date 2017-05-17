import nn_learn

import pickle
dat = pickle.load(open("data/TrainingData/pickled_generated_sets", 'rb'))
pca_dat = pickle.load(open("data/TrainingData/pickled_generated_sets_pca", 'rb'))

neural_net1_pca = {
    "num_features": 12,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [256, 256, 256],
    "use_dropout": True,
    "model_name": "model_pca_3_layers_256_lr1",
    "learning_rate": 1.0,
    "learning_rate_decay": 0.96,
    "metric_top_k": 1
}

neural_net1_non_pca = {
    "num_features": 136,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [2048, 2048, 2048],
    "use_dropout": True,
    "model_name": "model_non_pca_3_layers_2048",
    "learning_rate": 1.0,
    "learning_rate_decay": 0.96,
    "metric_top_k": 1
}

neural_net2_non_pca = {
    "num_features": 136,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [1024*4, 4096*2, 1024*4],
    "use_dropout": True,
    "model_name": "model_non_pca_3_layers_4096_8192_4096_lr0.3",
    "learning_rate": 0.2,
    "learning_rate_decay": 0.96,
    "metric_top_k": 1,
    "batch_size": 512
}


NN_3L_256_256_256 = nn_learn.NeuralNetwork(nn_learn.neural_net2_pca)

d = pca_dat

NN_3L_256_256_256.train(d["train_data"], d["train_labels"], d["valid_data"], d["valid_labels"], 600)

NN_3L_256_256_256.save()