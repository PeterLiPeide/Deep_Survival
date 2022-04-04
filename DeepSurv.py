# DeepSurv Realization with Pytorch


from turtle import forward
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F




class Image_Net(nn.Module):
    # Basic Deep Network for CT Scan / Imaging data 
    # Will be further integrated into our deep neural network model
    # To be replaced by our new network. Currently use an arbitrary CNN network

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.Maxpool2d(2, 2)
        self.fc1 = nn.Linear(2 * 6 * 5, 5)
        return

    def forward(self, x):
        x = self.pool(F.relu(self.conv1))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x


mymodel = Image_Net()



class DeepSurv_multimodal(Image_Net):

    def __init__(self, imgnet, n_SurvLayer, surv_struct, network_shape, activation = None, regularization = None):
        # Initiate 
        # Inputs: imgnet -- CNN network
        #         n_SurvLayer -- number of hidden layers in the CNN network
        #         surv_struct -- List of tuple (2 numbers each) giving input & output dimensions of the hidden layers 
        super().__init__()
        self.n_SurvLayer = n_SurvLayer
        self.surv_strcut = surv_struct
        self.activation = activation
        self.regularization = regularization
        self.shape = network_shape

        # Inherit network from imagenet
        self.imgnet = imgnet
        self.dp = nn.Dropout(p = 0.2)
        return

    def forward(self, X, img_x):
        # Define propogation for concated network
        hidden_img = self.imgnet(img_x)
        x_cate = np.concatenate(hidden_img, X, axis=0)
        for i in range(self.n_SurvLayer):
            x = nn.Linear(self.surv_struct[i][0], self.surv_strcut[i][1])
            if not self.activation:
                x = F.relu(self.dp(x))
            else:
                continue   # Leave for future 
        
        # Last layer with linear activation
        x = nn.Linear(self.surv_strcut[-1][-1], 1)
        
        return x


    def neg_log_likelihood(self, X, e, img_X):
        # Negative log likelihood function for survival risk prediciton 
        # Use as the loss function for network training 
        risk = self.forward(X, img_X)
        hazard_ratio = torch.tensor(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio))
        uncensored_likelihood = torch.transpose(risk) - log_risk
        censored_likelihood = uncensored_likelihood * e
        num_obs = torch.sum(e)
        neg_likelihood = -1 * torch.sum(censored_likelihood) / num_obs

        return neg_likelihood


    def train_network(self, X, e, img_X, x_init = None, lr = 0.001, max_epoch = 1000):
        # train the neural network
        if not x_init:
            x_init = torch.ones(self.shape)

        beta = x_init
    
        for i in range(max_epoch):
            loss = self.neg_log_likelihood(X, e, img_X)
            grad = loss.backward()
            beta -= lr * grad

        return beta


    pass
