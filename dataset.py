
import pandas as pd
import os
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms



DATA_DIR = 'C:/Users/kate/Desktop/WeCloudData/project/LandCover'
def metadata():
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    metadata = metadata[metadata['split']=='train']
    metadata = metadata.drop(columns=['split'])
    metadata['sat_image_path'] = metadata['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata['mask_path'] = metadata['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata = metadata.sample(frac=1).reset_index(drop=True)
    metadata_sample = metadata.iloc[:500]
    return metadata_sample

def classes():
    classes = pd.read_csv(os.path.join(DATA_DIR, "class_dict.csv"))
    label_names = classes['name'].tolist()
    label_rgb_values = classes[['r','g','b']].values.tolist()
    select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
    return label_names, label_rgb_values, select_classes


def one_hot_map(mask, label_rgb_values):
    """
    Convert RGB mask to a one-hot encoding image with shape (H, W, Channels)
    with Channels equal to the number of classes.
    
    from (2448, 2448, 3) to (2448, 2448, 7)
    """
    one_hot_map = []
    for colour in label_rgb_values:
        class_map = np.all(np.equal(mask, colour), axis = -1)
        one_hot_map.append(class_map)
    one_hot_map = np.stack(one_hot_map, axis=-1)
    
    # convert to int
    one_hot_map = one_hot_map.astype('float')

    return one_hot_map

def transpose_after_one_hot(x):
    return x.transpose(2, 0, 1)

def transpose_before_reverse_one_hot(x):
    return x.transpose(1, 2, 0)

def reverse_one_hot_map_cg(one_hot_map, label_rgb_values):
    """
    Convert one-hot encoded mask to an RGB image with shape (H, W, 3)
    using the provided label RGB values.
    """
    class_indices = np.argmax(one_hot_map, axis=-1)
    rgb_image = np.zeros_like(one_hot_map[..., 0:3])  # Assuming RGB is the first three channels
    for i, rgb_value in enumerate(label_rgb_values):
        mask = (class_indices == i)
        rgb_image[mask] = rgb_value
    return rgb_image

class LandCoverDataset(Dataset):
    def __init__(self, metadata, image_size, label_rgb_values=None, augmentation=None): #, transform=None):
        
        self.image_paths = metadata['sat_image_path'].tolist()
        self.mask_paths = metadata['mask_path'].tolist()
        self.image_size =image_size
        self.label_rgb_values = label_rgb_values
        self.augmentation = augmentation
        #self.augmented_size = len(self.image_paths) * (1 + (augmentation is not None))
        
    def __len__(self):
        return len(self.image_paths)
        #return self.augmented_size
        
    def __getitem__(self, index):
        # Load the image and its corresponding RGB mask
        img_path = self.image_paths[index]# % len(self.image_paths)]
        mask_path = self.mask_paths[index]# % len(self.mask_paths)]
        img = Image.open(img_path)
        mask_rgb = Image.open(mask_path)
        
        # Resize the image
        if self.image_size < 2048:
            img_resized = img.resize((self.image_size,self.image_size), resample=Image.BILINEAR)
            mask_resized = mask_rgb.resize((self.image_size,self.image_size), resample=Image.NEAREST)
        else:
            img_resized = img
            mask_resized = mask_rgb
        
        # Convert the image and the mask to NumPy arrays
        img_np = np.array(img_resized)
        mask_np = np.array(mask_resized)
        
        # Apply data augmentations to the image and the mask
        if self.augmentation:
        #if self.augmentation and index >= len(self.image_paths):
            augmented = self.augmentation(image=img_np, mask=mask_np)
            #img_np, mask_np = augmented['image'], mask_np = augmented['mask']
            img_np, mask_np = augmented['image'], augmented['mask']    

        # Convert the image and the mask to PyTorch tensors
        img_tensor = transforms.ToTensor()(img_np)
        mask_tensor = transforms.ToTensor()(mask_np).long()
    

        # Convert the RGB mask to one-hot encoding with shape (H, W, Channels)
        mask_rgb_one_hot = one_hot_map(mask_np, self.label_rgb_values)
        mask_one_hot_tensor = transpose_after_one_hot(mask_rgb_one_hot)

       
        print(f"   after transform shape: {img_tensor.shape}")

        return img_tensor, mask_one_hot_tensor
    import pandas as pd
import os
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms



DATA_DIR = 'C:/Users/kate/Desktop/WeCloudData/project/LandCover'
def metadata():
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    metadata = metadata[metadata['split']=='train']
    metadata = metadata.drop(columns=['split'])
    metadata['sat_image_path'] = metadata['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata['mask_path'] = metadata['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata = metadata.sample(frac=1).reset_index(drop=True)
    metadata_sample = metadata.iloc[:500]
    return metadata_sample

def classes():
    classes = pd.read_csv(os.path.join(DATA_DIR, "class_dict.csv"))
    label_names = classes['name'].tolist()
    label_rgb_values = classes[['r','g','b']].values.tolist()
    select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
    return label_names, label_rgb_values, select_classes


def one_hot_map(mask, label_rgb_values):
    """
    Convert RGB mask to a one-hot encoding image with shape (H, W, Channels)
    with Channels equal to the number of classes.
    
    from (2448, 2448, 3) to (2448, 2448, 7)
    """
    one_hot_map = []
    for colour in label_rgb_values:
        class_map = np.all(np.equal(mask, colour), axis = -1)
        one_hot_map.append(class_map)
    one_hot_map = np.stack(one_hot_map, axis=-1)
    
    # convert to int
    one_hot_map = one_hot_map.astype('float')

    return one_hot_map

def transpose_after_one_hot(x):
    return x.transpose(2, 0, 1)

def transpose_before_reverse_one_hot(x):
    return x.transpose(1, 2, 0)

def reverse_one_hot_map_cg(one_hot_map, label_rgb_values):
    """
    Convert one-hot encoded mask to an RGB image with shape (H, W, 3)
    using the provided label RGB values.
    """
    class_indices = np.argmax(one_hot_map, axis=-1)
    rgb_image = np.zeros_like(one_hot_map[..., 0:3])  # Assuming RGB is the first three channels
    for i, rgb_value in enumerate(label_rgb_values):
        mask = (class_indices == i)
        rgb_image[mask] = rgb_value
    return rgb_image

class LandCoverDataset(Dataset):
    def __init__(self, metadata, image_size, label_rgb_values=None, augmentation=None): #, transform=None):
        
        self.image_paths = metadata['sat_image_path'].tolist()
        self.mask_paths = metadata['mask_path'].tolist()
        self.image_size =image_size
        self.label_rgb_values = label_rgb_values
        self.augmentation = augmentation
        #self.augmented_size = len(self.image_paths) * (1 + (augmentation is not None))
        
    def __len__(self):
        return len(self.image_paths)
        #return self.augmented_size
        
    def __getitem__(self, index):
        # Load the image and its corresponding RGB mask
        img_path = self.image_paths[index]# % len(self.image_paths)]
        mask_path = self.mask_paths[index]# % len(self.mask_paths)]
        img = Image.open(img_path)
        mask_rgb = Image.open(mask_path)
        
        # Resize the image
        if self.image_size < 2048:
            img_resized = img.resize((self.image_size,self.image_size), resample=Image.BILINEAR)
            mask_resized = mask_rgb.resize((self.image_size,self.image_size), resample=Image.NEAREST)
        else:
            img_resized = img
            mask_resized = mask_rgb
        
        # Convert the image and the mask to NumPy arrays
        img_np = np.array(img_resized)
        mask_np = np.array(mask_resized)
        
        # Apply data augmentations to the image and the mask
        if self.augmentation:
        #if self.augmentation and index >= len(self.image_paths):
            augmented = self.augmentation(image=img_np, mask=mask_np)
            #img_np, mask_np = augmented['image'], mask_np = augmented['mask']
            img_np, mask_np = augmented['image'], augmented['mask']    

        # Convert the image and the mask to PyTorch tensors
        img_tensor = transforms.ToTensor()(img_np)
        mask_tensor = transforms.ToTensor()(mask_np).long()
    

        # Convert the RGB mask to one-hot encoding with shape (H, W, Channels)
        mask_rgb_one_hot = one_hot_map(mask_np, self.label_rgb_values)
        mask_one_hot_tensor = transpose_after_one_hot(mask_rgb_one_hot)

       
        print(f"   after transform shape: {img_tensor.shape}")

        return img_tensor, mask_one_hot_tensor
    		
