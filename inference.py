# Visualize the results or inference file
from PIL import Image
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from Autoencoder import ConvAutoencoder as convenc
from torchvision import datasets, transforms
import numpy as np
import os
import torch

#this class is used to get encoder output out of the autoencoder model by converting decoder layer to identity
class Identity(nn.Module): 
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



path='./filename.npy'
knn_path='./KNN.pkl'
auto_encoderpath='./final.pkl'
img_path='/content/drive/My Drive/dataset/100.jpg'




transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.Resize((224,224)),
    transforms.ToTensor()
    
])

imageData = Image.open(img_path).convert('RGB')
imageData = transform(imageData)

fileNames = np.load(path, mmap_mode='r')




model=convenc().cuda()

model_file=auto_encoderpath
model.load_state_dict(torch.load(model_file)) #load autoencoder model

#convert decoder layer to identity
model.t_conv1=Identity()
model.t_conv2=Identity()
model.t_conv3=Identity()


images=Variable(imageData.cuda())
output = model(images.unsqueeze(0))
# print(output.shape)
temp=torch.flatten(output)
# print("temp: ",temp.shape)
flattenImage=temp.to('cpu').detach().numpy()

with open(knn_path, 'rb') as f: #load KNN model
   model2=pickle.load(f)
_,index=model2.kneighbors([flattenImage])

# print(index)

rootpath= os.path.dirname(img_path)


#mapping image file name using fileNames array and index from KNN model

img1=os.path.join(rootpath,fileNames[index[0][0]][0]) 
img2=os.path.join(rootpath,fileNames[index[0][1]][0])
img3=os.path.join(rootpath,fileNames[index[0][2]][0])
img4=os.path.join(rootpath,fileNames[index[0][3]][0])
img5=os.path.join(rootpath,fileNames[index[0][4]][0])



imageData1 = Image.open(img1).convert('RGB')
imageData2 = Image.open(img2).convert('RGB')
imageData3 = Image.open(img3).convert('RGB')
imageData4 = Image.open(img4).convert('RGB')
imageData5 = Image.open(img5).convert('RGB')



imageData0=Image.open(img_path).convert('RGB')
fig=plt.figure(figsize=(3, 3))
plt.axis('off')
plt.title("Input Image")
plt.imshow(imageData0)



plot_image = np.concatenate((imageData1,imageData2, imageData3,imageData4,imageData5), axis=1)
fig=plt.figure(figsize=(12, 12))
plt.axis('off')
plt.title('Output Images')
plt.imshow(plot_image)
plt.show()


