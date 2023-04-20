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
import numpy as np

def train_cnn_stft(CONFIG, DEVICE=torch.device('cuda'), N_TRAIN=300, PROJECT_DIR='/root/SC1015-PROJECT/'):
    F_LOW = int(CONFIG['f_start'])
    F_HIGH = 400 + F_LOW
    
    model = ResNet10(input_size=(F_HIGH//5-F_LOW//5, 200) ,channels=CONFIG['input_channels'], dropout=CONFIG['dropout'])
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    train_id = tuple(sample([i for i in range(461)], N_TRAIN))
    test_id = tuple(set([i for i in range(461)])-set(train_id))
    
    trainset = set_b_dataclass(PROJECT_DIR, output_width=200, list_id=train_id, stft_low=F_LOW, stft_high=F_HIGH, oversample=True)
    trainloader = DataLoader(trainset, batch_size=int(CONFIG['batch_size']), shuffle=True)
    
    testset = set_b_dataclass(PROJECT_DIR, output_width=200, list_id=test_id, stft_low=F_LOW, stft_high=F_HIGH)
    testloader = DataLoader(testset, batch_size=int(CONFIG['batch_size']), shuffle=True)
    
    # epoch was determined by ray tune scheduler. Number 1000 below is to limit the ray tune scheduler.
    for _ in range(1000):
        val_metric = {
            'loss': 0,
            'accuracy_numerator': 0,
            'murmur_tpr_numerator': 0, 'murmur_tpr_denominator': 0,
            'extrastole_tpr_numerator': 0, 'extrastole_tpr_denominator': 0,
            'murmur_fpr_numerator': 0, 'murmur_fpr_denominator': 0,
            'extrastole_fpr_numerator': 0, 'extrastole_fpr_denominator': 0,
        }
        train_metric = {
            'loss': 0,
            'accuracy_numerator': 0, 
            'murmur_tpr_numerator': 0, 'murmur_tpr_denominator': 0,
            'extrastole_tpr_numerator': 0, 'extrastole_tpr_denominator': 0,
            'murmur_fpr_numerator': 0, 'murmur_fpr_denominator': 0,
            'extrastole_fpr_numerator': 0, 'extrastole_fpr_denominator': 0,
        }
        for i, data in enumerate(trainloader):
            stft, labels = data
            stft = stft.to(torch.float32).to(DEVICE)
            stft = stft.view(-1, 1, stft.size(1), stft.size(2))
            a = np.array(labels)
            one_hots = np.zeros((a.size, 3))
            one_hots[np.arange(a.size), a] = 1
            label_one_hot = torch.from_numpy(one_hots).to(torch.float32).to(DEVICE)
            model.train(True)
            optimizer.zero_grad()
            outputs_one_hot = model(stft)
            outputs_one_hot = nn.Softmax(dim=-1)(outputs_one_hot)
            loss = torch.nn.CrossEntropyLoss()(label_one_hot, outputs_one_hot)
            loss.backward()
            optimizer.step()
            train_metric['loss'] += loss.item()
            model.train(False)
            output_labels = torch.argmax(outputs_one_hot, 1)
            output_labels = output_labels.to('cpu')
            
            """Calculate metric [TPR, FPR]
            """
            train_metric['accuracy_numerator'] += torch.sum(output_labels==labels)
            train_metric['murmur_tpr_numerator'] += torch.sum(labels.ge(1.0) * labels.le(1.1) * output_labels.ge(1.0) * output_labels.le(1.1))
            train_metric['murmur_tpr_denominator'] += torch.sum(labels==torch.ones(size=(labels.size())) * 1)
            train_metric['murmur_fpr_numerator'] += torch.sum(output_labels==torch.ones(size=labels.size()) * (labels.lt(0.9) + labels.gt(1.5)))
            train_metric['murmur_fpr_denominator'] += torch.sum(labels!=torch.ones(size=(labels.size())) * 1)
            train_metric['extrastole_tpr_numerator'] += torch.sum(labels.gt(1.5) * output_labels.gt(1.5))
            train_metric['extrastole_tpr_denominator'] += torch.sum(labels==2*torch.ones(size=(labels.size())))
            train_metric['extrastole_fpr_numerator'] += torch.sum(labels.lt(1.5) * output_labels.gt(1.5))
            train_metric['extrastole_fpr_denominator'] += torch.sum(labels.ge(-1)) - torch.sum(labels.gt(1.5))
            
        for i, data in enumerate(testloader):
            stft, labels = data
            stft = stft.to(torch.float32).to(DEVICE)
            stft = stft.view(-1, 1, stft.size(1), stft.size(2))
            a = np.array(labels)
            one_hots = np.zeros((a.size, 3))
            one_hots[np.arange(a.size), a] = 1
            label_one_hot = torch.from_numpy(one_hots).to(torch.float32).to(DEVICE)
            outputs_one_hot = model(stft)
            outputs_one_hot = nn.Softmax(dim=-1)(outputs_one_hot)
            loss = torch.nn.CrossEntropyLoss()(label_one_hot, outputs_one_hot)
            val_metric['loss'] += loss.item()
            output_labels = torch.argmax(outputs_one_hot, 1)
            output_labels = output_labels.to('cpu')
            
            """Calculate metric [accuracy, TPR, FPR]
            """
            val_metric['accuracy_numerator'] += torch.sum(output_labels==labels)
            val_metric['murmur_tpr_numerator'] += torch.sum(labels.ge(1.0) * labels.le(1.1) * output_labels.ge(1.0) * output_labels.le(1.1))
            val_metric['murmur_tpr_denominator'] += torch.sum(labels==torch.ones(size=(labels.size())) * 1)
            val_metric['murmur_fpr_numerator'] += torch.sum(output_labels==torch.ones(size=labels.size()) * (labels.lt(0.9) + labels.gt(1.5)))
            val_metric['murmur_fpr_denominator'] += torch.sum(labels!=torch.ones(size=(labels.size())) * 1)
            val_metric['extrastole_tpr_numerator'] += torch.sum(labels.gt(1.5) * output_labels.gt(1.5))
            val_metric['extrastole_tpr_denominator'] += torch.sum(labels==2*torch.ones(size=(labels.size())))
            val_metric['extrastole_fpr_numerator'] += torch.sum(labels.lt(1.5) * output_labels.gt(1.5))
            val_metric['extrastole_fpr_denominator'] += torch.sum(labels.ge(-1)) - torch.sum(labels.gt(1.5))
            
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        
        session.report({ 
            "loss": float(train_metric['loss'] / trainset.__len__()),
            "accuracy": float(train_metric['accuracy_numerator']/trainset.__len__()),
            "murmur_tpr": float(train_metric['murmur_tpr_numerator'] / train_metric['murmur_tpr_denominator']),
            "murmur_fpr": float(train_metric['murmur_fpr_numerator']/ train_metric['murmur_fpr_denominator']),
            "extrastole_tpr": float(train_metric['extrastole_tpr_numerator'] / train_metric['extrastole_tpr_denominator']),
            "extrastole_fpr": float(train_metric['extrastole_fpr_numerator']/ train_metric['extrastole_fpr_denominator']),
            "murmur_val_tpr": float(val_metric['murmur_tpr_numerator'] / val_metric['murmur_tpr_denominator']),
            "murmur_val_fpr": float(val_metric['murmur_fpr_numerator']/ val_metric['murmur_fpr_denominator']),
            "val_loss": float(val_metric['loss'] / testset.__len__()),
            "val_accuracy": float(val_metric['accuracy_numerator'] / testset.__len__()), 
            "extrastole_val_tpr": float(val_metric['extrastole_tpr_numerator'] / val_metric['extrastole_tpr_denominator']),
            "extrastole_val_fpr": float(val_metric['extrastole_fpr_numerator']/ val_metric['extrastole_fpr_denominator']),
        }, checkpoint=checkpoint)
        