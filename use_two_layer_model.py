from dnn_model import*
from dnn_app_utils_v2 import *

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
print("**********2_layer_model Train Set**********")
predictions_train = predict(train_x, train_y, parameters)
print("**********2_layer_model Test Set**********")
predictions_test = predict(test_x, test_y, parameters)