from utils.model import ResNet10
from utils.dataset import set_b_dataset, set_b_dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from torch.utils.data import DataLoader
import os
from ray.air import session
from ray.air.checkpoint import Checkpoint

def train_cnn_stft(CONFIG, DEVICE=torch.device('cuda'), N_TRAIN=300, PROJECT_DIR='/root/SC1015-PROJECT/'):
    F_LOW = int(CONFIG['f_start'])
    # F_HIGH = F_LOW + int(CONFIG['f_range'])//16 * 16
    F_HIGH = 400 + F_LOW
    model = ResNet10(input_size=(F_HIGH//5-F_LOW//5, 200) ,channels=CONFIG['input_channels'], dropout=CONFIG['dropout'])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    train_id = tuple(sample([i for i in range(461)], N_TRAIN))
    test_id = tuple(set([i for i in range(461)])-set(train_id))
    # writer = SummaryWriter(f'runs/{F_LOW}_{F_HIGH}_Hz')
    
    trainset = set_b_dataclass(PROJECT_DIR, output_width=200, list_id=train_id, stft_low=F_LOW, stft_high=F_HIGH, oversample=True)
    trainloader = DataLoader(trainset, batch_size=int(CONFIG['batch_size']), shuffle=True)
    testset = set_b_dataclass(PROJECT_DIR, output_width=200, list_id=test_id, stft_low=F_LOW, stft_high=F_HIGH)
    testloader = DataLoader(testset, batch_size=int(CONFIG['batch_size']), shuffle=True)
    for _ in range(300):
        running_loss = 0
        running_val_loss = 0
        running_val_acc = 0
        running_acc = 0
        total_murmur_acc = 0
        running_murmur_acc = 0
        for i, data in enumerate(trainloader):
            stft, labels = data
            stft = stft.to(torch.float32).to(DEVICE)
            stft = stft.view(-1, 1, stft.size(1), stft.size(2))
            label_one_hot = F.one_hot(labels, num_classes=3).to(torch.float32).to(DEVICE)
            
            model.train(True)
            # grid = torchvision.utils.make_grid(stft)
            optimizer.zero_grad()
            outputs_one_hot = model(stft)
            outputs_one_hot = nn.Softmax()(outputs_one_hot)
            loss = torch.nn.CrossEntropyLoss()(label_one_hot, outputs_one_hot)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            model.train(False)
            
            output_labels = torch.argmax(outputs_one_hot, 1)
            output_labels = output_labels.to('cpu')
            running_acc += torch.sum(output_labels==labels) # line 43
        for i, data in enumerate(testloader):
            stft, labels = data
            stft = stft.to(torch.float32).to(DEVICE)
            stft = stft.view(-1, 1, stft.size(1), stft.size(2))
            label_one_hot = F.one_hot(labels, num_classes=3).to(torch.float32).to(DEVICE)
            outputs_one_hot = model(stft)
            outputs_one_hot = nn.Softmax()(outputs_one_hot)
            loss = torch.nn.CrossEntropyLoss()(label_one_hot, outputs_one_hot)
            running_val_loss += loss.item()
            output_labels = torch.argmax(outputs_one_hot, 1)
            output_labels = output_labels.to('cpu')
            running_val_acc += torch.sum(output_labels==labels)
            running_murmur_acc += torch.sum(labels.ge(1.0) * labels.le(1.1) * output_labels.ge(1.0) * output_labels.le(1.1))
            total_murmur_acc += torch.sum(labels==torch.ones(size=(labels.size())) * 1)
            
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        print(f"loss = {(running_val_loss / testset.__len__())} accuracy = {running_val_acc / testset.__len__()} ")
        session.report({
            "loss": (running_val_loss / testset.__len__()), 
            "accuracy": running_val_acc / testset.__len__(), 
            "murmur_true_positive": running_murmur_acc / testset.__len__()
            }, checkpoint=checkpoint)
        