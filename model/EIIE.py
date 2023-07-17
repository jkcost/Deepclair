import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EIIE(nn.Module):
    def __init__(self,batch_size,num_stock, in_features, window_len, hidden_dim=3,filter_num=10):
        super(EIIE, self).__init__()
        self.batch_size = batch_size
        self.num_stock = num_stock
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.filter_num = filter_num
        self.previous_w  = torch.zeros([batch_size, num_stock]).to(device)
        self.conv_layer = nn.Sequential(nn.Conv2d(  in_channels=in_features,
                                               out_channels=hidden_dim,#layer['filter_number']
                                               kernel_size=(1, 2), #layer['filter_shape']
                                               stride=[1,1], #layer['strides']
                                               padding=0),#layer['padding'])
                                         nn.ReLU(),)


        self.EIIE_Dense_layer = nn.Sequential(nn.Conv2d( in_channels=hidden_dim,  # you need to specify this according to your network structure
                                                    out_channels= filter_num, #layer["filter_number"]
                                                    kernel_size=(1, window_len-1),  # width is the width of the kernel
                                                    stride=(1, 1),
                                                    padding=0),nn.ReLU())



        self.EIIE_output_with = nn.Sequential(nn.Conv2d( in_channels=filter_num+1,  # you need to specify this according to your network structure
                                                    out_channels= 1, #layer["filter_number"]
                                                    kernel_size=(1,1),  # width is the width of the kernel
                                                    padding=0))

    def forward(self, X):
        """
          inputs: [batch, num_stock, window_len, num_features]
          mask: [batch, num_stock]
          outputs: [batch, scores]
          """
        X = X.permute(0,3,1,2)
        x = self.conv_layer(X)
        x = self.EIIE_Dense_layer(x)
        w = self.previous_w.reshape(x.shape[0],1,self.num_stock,1)
        x_w = torch.concat([x,w],axis=1)
        output_w = self.EIIE_output_with(x_w)[:,0,:,0]
        score = 1 / ((-output_w).exp() + 1)
        self.previous_w = score.detach()
        return score

    def reset(self,states):
        self.previous_w = torch.zeros([states.shape[0], self.num_stock]).to(device)



if __name__ == '__main__':
    a = torch.randn((37,26, 31, 5))
    net = EIIE(37,26,5, 31, 3)
    b = net(a)
    print(b)
