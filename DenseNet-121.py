import os 
import torch
import torchvision 
import torchvision.models as models 
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np 
import pydicom
from PIL import Image
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.densenet = models.densenet121(pretrained=True)
        self.fc_1 = nn.Linear(in_features=1000,out_features=512,bias=True)
        self.fc_2 = nn.Linear(in_features=512,out_features=128,bias=True)
        self.fc_3 = nn.Linear(in_features=128, out_features=2,bias=True)
        self.softmax = nn.Softmax()
    
    def forward(self,x):
        x = self.densenet(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.softmax(x)
        
        return x

net = Net()

data_path = '../input/'
train_images_path = os.path.join(data_path,'stage_2_train_images/')
test_images_path = os.path.join(data_path, 'stage_2_test_images/')
train_labels_path = os.path.join(data_path,'stage_2_train_labels.csv')

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]
)

class RSNADataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_labels = pd.read_csv(csv_file, engine='python').drop(['x','y','width','height'],axis=1).T.reset_index(drop=True).T
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.train_labels.iloc[idx, 0]) + '.dcm'
        image = pydicom.dcmread(img_name).pixel_array
        image = np.stack((image,)*3,axis=-1)
        image = Image.fromarray(image)
        label = self.train_labels.iloc[idx,1]

        if self.transform:
            image = self.transform(image)

        return image,label

def accuracy(outputs,labels):
    preds = outputs.max(dim=1)[1]
    return (preds==labels).float().sum()

# Use train_transform while training
def train(batch_size=16,epochs=5,transform=None):
    tfmd_dataset = RSNADataset(csv_file=train_labels_path,root_dir=train_images_path,
        transform=transform)
    dataloader = DataLoader(tfmd_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    if torch.cuda.is_available():
        net.cuda()
        nn.DataParallel(net)
        print("GPU")
    epoch_loss_data = []
    epoch_accuracy_data = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    
    for epoch in range(1,epochs+1):  # loop over the dataset multiple times

        running_loss, running_loss_total,epoch_accuracy = 0.0, 0.0,0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            epoch_accuracy += accuracy(outputs,labels)
            running_epoch_acc = epoch_accuracy/(labels.size()[0] * (i+1))
            running_loss += loss.item()
            running_loss_total += loss.item()/(labels.size()[0] * (i+1))
            if i%30 == 29:
                print('[%d, %5d] running loss: %.3f, epoch_accuracy: %f' % (epoch, i + 1,running_loss_total,running_epoch_acc))
                running_loss = 0.0
            if i%100 == 99:
                torch.save(net.state_dict(), 'checkpoint')
                torch.save(optimizer.state_dict(), 'optimizer_checkpoint')
                with open('Running_loss.p','wb') as f:
                    pickle.dump(running_loss_total,f)
                with open('Running_accuracy.p','wb') as f:
                    pickle.dump(running_epoch_acc,f)
        epoch_accuracy_data.append(running_epoch_acc)
        epoch_loss_data.append(running_loss_total)
        
        return data,outputs

train(transform=train_transform,epochs=0,batch_size=16)