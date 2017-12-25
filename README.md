# DNN-Application
Deep Neural Network for Image Classification: Application

*Helper functions as flows:*  
*[Building deep neural network step by step](http://yuehua.me/deep%20learning/2017/12/21/Building-your-Deep-Neural-Network-Step-by-Step/)*

## Two-layer neural network 
### Architecture  
<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/2layerNN_kiank.png" style="width:500px;height:300px;">
</div>  

### Helper Functions   
```python
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

### Results  
<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/2-l-iter.png" style="width:600px;height:400px;">
</div>  


## L-layer Neural Network  
### Architecture  
<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/LlayerNN_kiank.png" style="width:600px;height:400px;">
</div>  

### Helper Functions   
```python
def initialize_parameters_deep(layer_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```  

### Results  
<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/l-l-iter.png" style="width:600px;height:400px;">
</div>  

## Reference
1.[Deep Learning](https://www.deeplearning.ai/)  
2.[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/) 