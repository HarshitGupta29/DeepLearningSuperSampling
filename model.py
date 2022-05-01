import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = torch.Conv2d(out_channels=64, kernel_size=(3,3), stride=(1,1), act=None, padding='SAME', W_init=W_init)
        self.residual_block = self.make_layer()
        self.conv2 = torch.Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
                            b_init=None)
        self.bn1 = torch.BatchNorm2d(act=None,  gamma_init=G_init)
        self.upsample1 = torch.UpSampling2d(scale=(2,2), method='bilinear')
        self.conv3 = torch.Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
                            b_init=None)
        self.bn2 = torch.BatchNorm2d(act= torch.ReLU, gamma_init=G_init)
        self.upsample2 = torch.UpSampling2d(scale=(4,4),method='bilinear')
        self.conv4 = torch.Conv2d(out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
                            b_init=None)
        self.bn3 = torch.BatchNorm2d(act = torch.ReLU, gamma_init=G_init)
        self.conv5 = torch.Conv2d(out_channels=3, kernel_size=(1,1), stride=(1,1), act = torch.Tanh, padding='SAME', W_init=W_init)


    def forward(self, x):
        x = self.conv1(x)
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.upsample2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        return x



class Discriminator(torch.nn.Module):
    def __init__(self, image_size: int, kernel_size: int, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.filter_sizes = [64,128,128,256,256,512,512]
        self.strides = [2,1,2,1,1,1,1]
        model_list = self.block(64,1, self.device)
        for i, j in zip(self.filter_sizes, self.strides):
            model_list.extend(self.block(i,j))
        self.conv1 = torch.Conv2d(out_channels=64, kernel_size=(3,3), stride=(1,1), padding='SAME', W_init=W_init)
        self.residual_block = self.make_layer()
        self.conv2  = torch.Conv2d(out_channels=64, kernel_size=(3,3), stride=(1,1),padding='SAME', W_init=W_init, b_init = None)
        self.bn1 = torch.BatchNorm2d(num_features=64, act=None)
        self.conv3 = torch.Conv2d(out_channels=256, kernel_size=(3,3), stride=(1,1),padding='SAME', W_init = W_init)
        self.conv4 = torch.Conv2d(out_channels=256, kernel_size=(3,3), stride=(1,1), padding='SAME', W_init=W_init) 
        self.conv5 = torch.Conv2d(3, kernel_size=(1,1), stride=(1,1), padding='SAME', W_init=W_init)

    def forward(self, x):
        x = self.conv1(x)
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp
        x = self.conv3(x)
        x = self.subpiexlconv1(x)
        x = self.conv4(x)
        x = self.subpiexlconv2(x)
        x = self.conv5(x)

        return x
        


    def block(self, filter_size, strides):
        first = torch.nn.Conv2d(in_channels=3, out_channels=filter_size, kernel_size=self.kernel_size,device = self.device)
        second = torch.nn.LeakyReLU(0.2)
        return [first, second]


class dlss(torch.nn.Module):
    def __init__(self) -> None:
        super(dlss, self).__init__()
        self.conv1 = torch.Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv2 = torch.Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.maxpool1 = torch.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME')

        self.conv3 = torch.Conv2d(out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv4 = torch.Conv2d(out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.maxpool2 = torch.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME')

        self.conv5 = torch.Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv6 = torch.Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv7 = torch.Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv8 = torch.Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.maxpool3 = torch.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME')

        self.conv9 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv10 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv11 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv12 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.maxpool4 = torch.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME') # (batch_size, 14, 14, 512)

        self.conv13 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv14 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv15 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.conv16 = torch.Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=torch.ReLU, padding='SAME')
        self.maxpool5 = torch.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME') # (batch_size, 7, 7, 512)

        self.flat = torch.Flatten()
        self.dense1 = torch.Linear(out_features=4096, act=torch.ReLU)
        self.dense2 = torch.Linear(out_features=4096, act=torch.ReLU)
        self.dense3 = torch.Linear(out_features=1000, act=torch.identity)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool3(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool4(x)
        conv = x
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool5(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x, conv

