#import mlflow
import torch
from aws_dataset import metadata, classes
from train import get_loader, train_function
#from metric import accuracy
from UNet_model import UNet
import argparse
import torch.nn as nn
import albumentations as A
import datetime
import io
#from io import BytesIO
import boto3

data_augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
   ])

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args(from_jupyter = True):
    if from_jupyter:
        args = DotDict()
        args.image_size = 512
        args.accum_step = 5
        #args.use_albumentation = data_augmenter
        args.batch_size = 3
        args.valid_batch_size = 2
        args.test_batch_size = 1
        args.epochs = 4 #10
        args.lr = 0.0001
        args.num_workers = 0
        args.test_num_workers = 0
        args.log_interval = 10
        return args
    else:
        """ parse command line arguments if running from command line, like `python landcover.py` """
        parser = argparse.ArgumentParser(description = 'Train image segmentation')
        parser.add_argument(
            '--image_size', type=int, default=512, metavar='N',
            help='size of the image'
        )
        parser.add_argument(
            '--accum_step', type=int, default=10, metavar='N',
            help='accumulation step: number of batches (default-10)'
        )
        
        #parser.add_argument(
        #    '--use_albumentation', default=data_augmenter, type=bool,
        #    help='use_albumentation'
        #)

        parser.add_argument(
            '--batch_size', type=int, default=3, metavar='N',
            help='input batch size for training (default:3)'
        )

        parser.add_argument(
            '--test_batch_size', type=int, default=1, metavar='N',
            help='input batch size for testing (default:1)'
        )
        
        parser.add_argument(
            '--valid_batch_size', type=int, default=1, metavar='N',
            help='input batch size for validation (default:1)'
        )

        parser.add_argument(
            '--epochs', type=int, default=3, metavar='N',
            help='number of epochs to train (default:10)'
        )

        parser.add_argument(
            '--lr', type=float, default=0.0001, metavar='LR',
            help='learning rate (default:0.001)'
        )

        parser.add_argument(
            '--num_workers', type=int, default=2,
            help='number of workers to load data for training (default:2)'
        )

        parser.add_argument(
            '--test_num_workers', type=int, default=1,
            help='number of workers to load data for testing (default:1)'
        )
        parser.add_argument(
            '--log_interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status'
        )
        args = parser.parse_args()
        return args
    
args = parse_args(from_jupyter = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metadata_sample = metadata()
label_names, label_rgb_values, select_classes = classes()

train_loader, test_loader, val_loader = get_loader(metadata_sample, label_rgb_values,image_size=args.image_size, batch_size=args.batch_size, valid_batch_size=args.valid_batch_size, test_batch_size=args.test_batch_size, num_workers=args.num_workers, test_num_workers=args.test_num_workers, augmentation=True)

unet = UNet(in_channels = 3, out_channels = 7, features = 64)
unet.to(device)

#define optimizer
optimizer = torch.optim.Adam([
    dict(params = unet.parameters(), lr = args.lr),
])

#define loss function
loss_fn = nn.CrossEntropyLoss()

y, loss_val_values, accuracies, accuracies_val \
    = train_function(data = train_loader,
                     data_val = val_loader,
                     model = unet,
                     optimizer = optimizer,
                     device = device,
                     loss_fn = loss_fn,
                     accum_step=args.accum_step,
                     epochs = args.epochs,
                     lr = args.lr,
                     image_size=args.image_size,
                     batch_size=args.batch_size,
                     accumulation = False)

now = datetime.datetime.now()
file_name = f"LandCover/model/UNET_{now.strftime('%Y%m%d-%H%M')}_512_full_epoch_10.pth"
buffer = io.BytesIO()
torch.save(unet.state_dict(), buffer)
s3 = boto3.client('s3')
s3.put_object(Bucket="mlops-deploy", Key=file_name, Body=buffer.getvalue())  