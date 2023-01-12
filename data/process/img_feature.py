'''
   Extract image feature from ResNet 101 model
'''
import os 
import json
import pickle
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet101(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        # output = self.net.avgpool(output)
        return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image feature from ResNet 101')
    parser.add_argument('input', help='image folder')
    parser.add_argument('output', help='output folder')
    args = parser.parse_args()

    model = net()
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    datas = {}
    for image in os.listdir(args.input):
        img_path = os.path.join(args.input, image)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        #print('Image Size ', img.shape)
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        x = x.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y.cpu())
        y = y.reshape(y.size(0), -1)
        y = y.data.numpy()
        # print('Feature ', y.shape)
        datas[image] = y
        print('Total images ', len(datas))
        #break
    pickle.dump(datas, open(args.output, 'wb'))
    
    



        


