import torch
from PIL import Image
from picamera2 import Picamera2, Preview
from libcamera import Transform
import torchvision.transforms.v2 as v2
import time
from torch import nn

# Define the model
class Net(nn.Module):
        def __init__(self,input_dim= 3*50*50,output_dim=43):

            super(Net,self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.metrics = {}
            self.flatten = nn.Flatten()
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
            self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
            self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
            self.batchnorm2 = nn.BatchNorm2d(256)
            self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3)
            self.conv6 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3)
            self.batchnorm3 = nn.BatchNorm2d(1024)
   
            self.l1 = nn.Linear(1024*4*4,512)
            self.l2 = nn.Linear(512,128)
            self.batchnorm4 = nn.LayerNorm(128)
            self.l3 = nn.Linear(128,output_dim)
        
        
        def forward(self,input):
            
            conv = self.conv1(input)
            conv = self.conv2(conv)
            batchnorm = self.relu(self.batchnorm1(conv))
            maxpool = self.maxpool(batchnorm)

            conv = self.conv3(maxpool)
            conv = self.conv4(conv)
            batchnorm = self.relu(self.batchnorm2(conv))
            maxpool = self.maxpool(batchnorm)

            conv = self.conv5(maxpool)
            conv = self.conv6(conv)
            batchnorm = self.relu(self.batchnorm3(conv))
            maxpool = self.maxpool(batchnorm)
            flatten = self.flatten(maxpool)
            
            dense_l1 = self.l1(flatten)
            dropout = self.dropout3(dense_l1)
            dense_l2 = self.l2(dropout)
            batchnorm = self.batchnorm4(dense_l2)
            dropout = self.dropout2(batchnorm)
            output = self.l3(dropout)
            
            return output



# Load the trained model
model = Net()
model.load_state_dict(torch.load('final_model.pth'))
model.eval()

# Define the image transformation
transform = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize([50,50]),
        v2.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

# Initialize PiCamera and capture an image
camera = Picamera2()
camera.preview_configuration.sensor.output_size = (800, 600)
camera.preview_configuration.main.size = (800,600)
camera.start_and_capture_file(name = 'testimg.jpg', delay = 5)


# Load and preprocess the image
image = Image.open('testimg.jpg')
image = transform(image).unsqueeze(0)

# Make predictions
with torch.no_grad():
    output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
#Defining the class dict
class_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


# Process the output
score, predicted = torch.max(probabilities, 1)
predicted_class_name = class_names[predicted.item()]
conf = score.item()*100
print('Predicted class:', predicted_class_name)
print('Confidence score:', round(conf), '%' )
