import nn_learn

import pickle

import base_model

dat = pickle.load(open("data/TrainingData/pickled_generated_sets", 'rb'))
pca_dat = pickle.load(open("data/TrainingData/pickled_generated_sets_pca", 'rb'))



model = nn_learn.NeuralNetwork(nn_learn.neural_net2_pca_lr03)

d = pca_dat

model.train(d["train_data"], d["train_labels"], d["valid_data"], d["valid_labels"], 600)

model.save()

print("Training Accuracy: {}".format(base_model.accuracy(model, pca_dat["train_data"], pca_dat["train_labels"])))
print("Accuracy:", base_model.accuracy(model, pca_dat["valid_data"], pca_dat["valid_labels"]))
print("Precision:", base_model.precision(model, pca_dat["valid_data"], pca_dat["valid_labels"]))
print("Recall:", base_model.recall(model, pca_dat["valid_data"], pca_dat["valid_labels"]))
print("Confusion Matrix:", base_model.confusion_matrix(model, pca_dat["valid_data"], pca_dat["valid_labels"]))

with open ("data/Models/"+model.params["model_name"]+"_props.txt","w") as f:
    f.write("Training Accuracy: {}\n".format(base_model.accuracy(model, pca_dat["train_data"], pca_dat["train_labels"])))
    f.write("Accuracy: {}\n".format(base_model.accuracy(model, pca_dat["valid_data"], pca_dat["valid_labels"])))
    f.write("Precision: {}\n".format( base_model.precision(model, pca_dat["valid_data"], pca_dat["valid_labels"])))
    f.write("Recall: {}\n".format( base_model.recall(model, pca_dat["valid_data"], pca_dat["valid_labels"])))
    f.write("Confusion Matrix:\n {}\n".format(base_model.confusion_matrix(model, pca_dat["valid_data"], pca_dat["valid_labels"])))