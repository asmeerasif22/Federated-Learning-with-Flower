import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from tqdm import tqdm
from PIL import Image
from skimage import exposure
from sklearn.model_selection import train_test_split
from flwr_datasets import FederatedDataset

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 20

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

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        correct = 0
        total = 0
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def prepare_dataset():
    # Initialize FederatedDataset with the desired dataset
    fds = FederatedDataset(dataset="dpdl-benchmark/gtsrb", partitioners={"train": NUM_CLIENTS})
    img_key = "image"  
    pytorch_transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize([50,50]),
        v2.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))

    ])
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch
    
    trainsets = []
    validsets = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=42)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])

    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)

    return trainsets, validsets, testset

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model 
        self.model = Net()
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0004, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=8)
        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)
        #Saving the model
        torch.save(self.model.state_dict(), "final_model.pth")
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

def main():
    args = parser.parse_args()
    client_id = args.cid
    print(args)
    assert client_id < NUM_CLIENTS

    # Load the dataset
    trainsets, valsets, _ = prepare_dataset()

    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid]
        ).to_client(),
    )

if __name__ == "__main__":
    main()