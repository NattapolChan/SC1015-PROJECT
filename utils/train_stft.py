from utils.model import ResNet10, InceptionTime
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
import sklearn
import sklearn.metrics as metrics

def fit_cnn_stft(CONFIG, DEVICE=torch.device('cuda:1'), N_TRAIN=380, 
                   PROJECT_DIR='/root/SC1015-PROJECT/', model_name='InceptionTime'):
    F_LOW = int(CONFIG['f_start'])
    F_HIGH = 100 + F_LOW
    
    if model_name=='ResNet':
        model = ResNet10(channels=CONFIG['input_channels'], 
                         dropout=CONFIG['dropout'])
    elif model_name=='InceptionTime':
        model = InceptionTime((20, 104), 3)
    else :
        raise("Model Name Error")
        
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    train_id = tuple(sample([i for i in range(461)], N_TRAIN))
    test_id = tuple(set([i for i in range(461)])-set(train_id))
    
    trainset = set_b_dataclass(PROJECT_DIR, output_width=104, list_id=train_id, 
                               stft_low=F_LOW, stft_high=F_HIGH, 
                               out_classes = list(('normal', 'extrastole', 'murmur')), oversample=True)
    trainloader = DataLoader(trainset, batch_size=int(CONFIG['batch_size']), shuffle=True)
    
    testset = set_b_dataclass(PROJECT_DIR, output_width=104, list_id=test_id, 
                              stft_low=F_LOW, out_classes = list(('normal', 'extrastole', 'murmur')), stft_high=F_HIGH)
    testloader = DataLoader(testset, batch_size=int(CONFIG['batch_size']), shuffle=True)
    
    for _ in range(30):
        list_y_pred, list_y_label = np.array([]), np.array([])
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
            outputs_one_hot = outputs_one_hot.to('cpu')
            output_labels = torch.argmax(outputs_one_hot, 1)
            
            """Calculate metric []
            """
            train_metric['accuracy_numerator'] += torch.sum(output_labels==labels)
            train_metric['murmur_tpr_numerator'] += torch.sum((labels==1) * (output_labels==1))
            train_metric['murmur_tpr_denominator'] += torch.sum(labels==1)
            train_metric['extrastole_tpr_numerator'] += torch.sum((labels==2) * (output_labels==2))
            train_metric['extrastole_tpr_denominator'] += torch.sum(labels==2)
            
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
            outputs_one_hot = outputs_one_hot.to('cpu')
            output_labels = torch.argmax(outputs_one_hot, 1)
            
            """Calculate metric [accuracy, TPR, FPR]
            """
            val_metric['accuracy_numerator'] += torch.sum(output_labels==labels)
            val_metric['murmur_tpr_numerator'] += torch.sum((labels==1) * (output_labels==1))
            val_metric['murmur_tpr_denominator'] += torch.sum(labels==1)
            val_metric['extrastole_tpr_numerator'] += torch.sum((labels==2) * (output_labels==2))
            val_metric['extrastole_tpr_denominator'] += torch.sum(labels==2)
            list_y_pred = np.append(list_y_pred, output_labels.numpy())
            list_y_label = np.append(list_y_label, labels.numpy())
            
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        
        session.report({ 
            "loss": float(train_metric['loss'] / trainset.__len__()),
            "accuracy": float(train_metric['accuracy_numerator']/trainset.__len__()),
            "g_mean": float(train_metric['murmur_tpr_numerator'] / train_metric['murmur_tpr_denominator']) \
                * float(train_metric['extrastole_tpr_numerator'] / train_metric['extrastole_tpr_denominator']),
            "murmur_tpr": float(train_metric['murmur_tpr_numerator'] / train_metric['murmur_tpr_denominator']),
            "extrastole_tpr": float(train_metric['extrastole_tpr_numerator'] / train_metric['extrastole_tpr_denominator']),
            "val_loss": float(val_metric['loss'] / testset.__len__()),
            "val_accuracy": float(val_metric['accuracy_numerator'] / testset.__len__()), 
            "murmur_val_tpr": float(val_metric['murmur_tpr_numerator'] / val_metric['murmur_tpr_denominator']),
            "extrastole_val_tpr": float(val_metric['extrastole_tpr_numerator'] / val_metric['extrastole_tpr_denominator']),
        }, checkpoint=checkpoint)
        class_name = ['normal', 'murmur', 'extrastole']
        if _==5:
            print(f'Frequency from {F_LOW} to {F_LOW + 100}')
            print(metrics.classification_report(list_y_label, list_y_pred, target_names=class_name))
        
