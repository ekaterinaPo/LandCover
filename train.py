from sklearn.model_selection import train_test_split
import albumentations as A
from aws_dataset import LandCoverDataset
import argparse
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import mlflow
from UNet_model import UNet
from metric import accuracy
from tqdm import tqdm
import numpy as np

def get_loader(metadata_sample, label_rgb_values, image_size, batch_size, valid_batch_size, test_batch_size, num_workers, test_num_workers, augmentation: bool):
    train_ids, test_ids = train_test_split(
        metadata_sample, 
        train_size=0.8,
        shuffle=True, random_state=42
    )
    test_ids, val_ids = train_test_split(test_ids, 
        train_size=0.5, 
        random_state=42
    )
    train_dataset = LandCoverDataset(
        train_ids,
        image_size = image_size,
        label_rgb_values=label_rgb_values,   
    )
    test_dataset = LandCoverDataset(
        test_ids,
        image_size = image_size,
        label_rgb_values=label_rgb_values,
    )
    val_dataset = LandCoverDataset(
        val_ids,
        image_size = image_size,
        label_rgb_values=label_rgb_values,
    )
    if augmentation is True:
        data_augmenter = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        augmented_train_dataset = LandCoverDataset(
            train_ids,
            image_size = image_size,
            label_rgb_values=label_rgb_values,
            augmentation = data_augmenter,     
        )
        augmented_val_dataset = LandCoverDataset(
            val_ids,
            image_size = image_size,
            label_rgb_values=label_rgb_values,
            augmentation = data_augmenter,
        )
        final_train_dataset = ConcatDataset([train_dataset, augmented_train_dataset])
        final_val_dataset = ConcatDataset([val_dataset, augmented_val_dataset])
    else:

        final_train_dataset = train_dataset
        final_val_dataset = val_dataset
    
    train_loader = DataLoader(final_train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, 
                             batch_size = test_batch_size, #len(test_dataset),
                             shuffle = False, num_workers=test_num_workers)
    val_loader = DataLoader(final_val_dataset, 
                             batch_size = valid_batch_size,#len(final_val_dataset), 
                             shuffle = False, num_workers=test_num_workers)
    return train_loader, test_loader, val_loader


def train_function(data, data_val, model, optimizer, device, loss_fn, accum_step, epochs, lr, image_size, batch_size, accumulation: bool):
    print('Entering into train function')

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow_running = True
    try:
        mlflow.start_run()
    except mlflow.exceptions.MlflowException:
        mlflow_running = False

    if mlflow_running:
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("accumulation", accumulation)
    loss_values = [] 
    loss_val_values = []
    accuracies = []
    accuracies_val = []
    
    model.to(device)

    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(data)): #for i, batch in enumerate(tqdm(data)):
            X, y = batch 
            X, y = X.to(device), y.to(device)
            #forward pass
            preds = model(X)
            loss = loss_fn(preds, y) #compute loss function
            if accumulation is True:
                loss = loss/accum_step #normalize loss to account for batch accumulation
            loss_values.append(loss.item())
            acc = accuracy(preds, y, 7)
            accuracies.append(acc)
            loss.backward()#backward pass
            
            if accumulation is True:
                if ((i+1) % accum_step == 0) or (i+1 ==len(data)): #wait
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
                    
            if mlflow_running:
                mlflow.log_metric("accuracy_train", acc)
                mlflow.log_metric("loss_train", loss.item())

        print("Starting validation")
        cur_accuracies_val = []
        cur_loss_val_values = []
        for index_val, batch_val in enumerate(data_val):
            X_val, y_val = batch_val
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            preds_val = model(X_val)
            loss_val = loss_fn(preds_val, y_val)
            cur_loss_val_values.append(loss_val.item())
            cur_accuracies_val.append(accuracy(preds_val, y_val, 7))

        if mlflow_running:
            mlflow.log_metric("accuracy_train", acc)
            mlflow.log_metric("loss_train", loss.item())
        #loss_val_values.append(loss_val.item())
        cur_loss_val_values = np.mean(cur_loss_val_values)
        cur_accuracies_val = np.mean(cur_accuracies_val)
        loss_val_values.append(cur_loss_val_values)
        accuracies_val.append(cur_accuracies_val)
        if mlflow_running:
            mlflow.log_metric("accuracy_valid", cur_loss_val_values)
            mlflow.log_metric("loss_valid", cur_accuracies_val)
        print("Finished with validation")

    return loss_values, loss_val_values, accuracies, accuracies_val


