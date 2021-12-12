import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def train(model, optimizer, loss, train_loader, epochs=100, scheduler=None, valid_loader=None, gpu=None):
    # GPU
    if gpu is not None:
        model = model.cuda(gpu)

    epochs_train_loss = []
    epochs_valid_loss = []
    epochs_train_acc = []
    epochs_valid_acc = []
    for ep in range(epochs):
        begin = time.time()
        model.training = True
        
        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(train_loader):
            # GPU
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)
            
            predictions = model(inputs).squeeze(dim=1)
            err = loss(predictions, targets)

            # Machine is learning
            err.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            labels = (F.sigmoid(predictions) >= 0.5) * 1
            
            # Clean GPU
            if gpu is not None:
                err = err.detach().cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                labels = labels.cpu()
                torch.cuda.empty_cache()
            
            all_losses.append(err)
            all_predictions.append(labels.unsqueeze(-1))
            all_targets.append(targets.unsqueeze(-1))
            accuracy_batch = accuracy_score(targets, labels)
            
            print(f'\rBatch : {i+1} / {len(train_loader)} - Accuracy : {accuracy_batch*100:.2f}% - Loss : {err:.2e}', end='')
        
        all_predictions = torch.vstack(all_predictions)
        all_targets = torch.vstack(all_targets)
        
        train_loss = np.vstack(all_losses).mean()
        train_acc = accuracy_score(all_targets, all_predictions)
        
        # Historique
        epochs_train_acc.append(train_acc)
        epochs_train_loss.append(train_loss)
        
        if scheduler is not None:
            scheduler.step()
        
        # Validation step
        if valid_loader is not None:
            valid_loss, valid_acc = valid(model, loss, valid_loader, gpu)
            # Historique
            epochs_valid_acc.append(valid_acc)
            epochs_valid_loss.append(valid_loss)
            end = time.time()
            print(f'\rEpoch : {ep+1} - Train Accuracy : {train_acc*100:.2f}% - Train Loss : {train_loss:.2e} - Valid Accuracy : {valid_acc*100:.2f}% - Valid Loss : {valid_loss:.2e} - Time : {end - begin:.2f} sec')
        else:
            # Afficher les informations de l’époque
            end = time.time()
            print(f'\rEpoch : {ep+1} - Train Accuracy : {train_acc*100:.2f}%  - Train Loss : {train_loss:.2e} - Time : {end - begin:.2f} sec')
        
    if valid_loader is not None:
        return epochs_train_acc, epochs_train_loss, epochs_valid_acc, epochs_valid_loss
    
    return epochs_train_acc, epochs_train_loss


def valid(model, loss, valid_loader, gpu):
    model.training = False
    with torch.no_grad():
        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(valid_loader):
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze(dim=1)
            err = loss(predictions, targets)

            all_losses.append(err.detach().cpu())
            
            labels = (F.sigmoid(predictions) >= 0.5) * 1
            # Clean GPU
            if gpu is not None:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                labels = labels.cpu()
                torch.cuda.empty_cache()
                
            all_predictions.append(labels.unsqueeze(-1))
            all_targets.append(targets.unsqueeze(-1))
            
            print(f'\rValid batch : {i+1} / {len(valid_loader)}', end='')
        
        all_losses = torch.vstack(all_losses)
        all_predictions = torch.vstack(all_predictions)
        all_targets = torch.vstack(all_targets)
        valid_acc = accuracy_score(all_targets, all_predictions)
        
        return all_losses.mean(), valid_acc


def test(model, loss, test_loader, gpu):
    model.training = False
    with torch.no_grad():
        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(test_loader):
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze(dim=1)
            err = loss(predictions, targets)

            all_losses.append(err.detach().cpu())

            # Clean GPU
            if gpu is not None:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                torch.cuda.empty_cache()
                
            all_predictions.append(((F.sigmoid(predictions) >= 0.5) * 1).unsqueeze(-1))
            all_targets.append(targets.unsqueeze(-1))
            
            print(f'\rTest batch : {i+1} / {len(test_loader)}', end='')
            
        all_losses = torch.vstack(all_losses)
        all_predictions = torch.vstack(all_predictions)
        all_targets = torch.vstack(all_targets)
        test_acc = accuracy_score(all_targets, all_predictions)
        
        return all_losses.mean(), test_acc


def predict(model, dataloader=None, tensor_data=None, gpu=None):
    model.training = False
    
    if gpu is not None:
        model = model.cuda(gpu)
    
    if dataloader is not None:
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if gpu is not None:
                    inputs = inputs.cuda(gpu)
                    targets = targets.float().cuda(gpu)

                predictions = model(inputs).squeeze(dim=1)

                # Clean GPU
                if gpu is not None:
                    inputs = inputs.cpu()
                    targets = targets.cpu()
                    predictions = predictions.cpu()
                    torch.cuda.empty_cache()

                all_predictions.append(((F.sigmoid(predictions) >= 0.5) * 1).unsqueeze(-1))
                all_targets.append(targets.unsqueeze(-1))

                print(f'\rPredict batch : {i+1} / {len(dataloader)}', end='')

            all_predictions = torch.vstack(all_predictions)
            all_targets = torch.vstack(all_targets)
        return all_predictions, all_targets
    
    if tensor_data is not None:
        if gpu is not None:
            tensor_data = tensor_data.cuda(gpu)
        
        with torch.no_grad():
            predictions = model(tensor_data).squeeze()
        return (F.sigmoid(predictions) >= 0.5) * 1
    
    return None
