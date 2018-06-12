from dnn_model import*
from dnn_app_utils_v2 import *

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
print("**********l_layer_model Train Set**********")
pred_train = predict(train_x, train_y, parameters)
print("**********l_layer_model Test Set**********")
pred_test = predict(test_x, test_y, parameters)

#Results Analysis
print("Results Analysis:")
print_mislabeled_images(classes, test_x, test_y, pred_test) 