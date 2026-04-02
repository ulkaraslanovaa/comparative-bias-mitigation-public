'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from glob import glob
from PIL import Image
import pandas as pd

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

class CMNISTDataset(Dataset):
    def __init__(self,root,split,transform=None, image_path_list=None):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*"))            
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root

        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


class WaterbirdsDataset(Dataset):

    def __init__(self, root: str, split: str, transform=None, balanced=False):
        self.root = root
        self.data_dir = os.path.join(self.root, "data")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2
        }
        
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        df = df[df["split"] == split_map[split]].reset_index(drop=True)
        self.df = df

        if split == "train" and balanced:
            self.df = pd.read_csv(os.path.join(self.root, "train_balanced.csv")).reset_index(drop=True)
        
        self.fname_col = "unique_img_filename"

        self.paths  = [os.path.join(self.data_dir, fn) for fn in self.df[self.fname_col].tolist()]
        self.labels = self.df["y"].astype(int).to_numpy()
        self.biases = self.df["place"].astype(int).to_numpy()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        attr = torch.tensor([int(self.labels[idx]), int(self.biases[idx])], dtype=torch.long)
        return img, attr, path

class FairfaceDataset(Dataset):

    def __init__(self, root: str, split: str, transform=None):
        self.root = root
        self.data_dir = os.path.join(self.root, "data")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2
        }
        
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        df = df[df["split"] == split_map[split]].reset_index(drop=True)
        self.df = df

        self.fname_col = "file"

        self.paths  = [os.path.join(self.data_dir, fn) for fn in self.df[self.fname_col].tolist()]
        self.labels = self.df["gender"].astype(int).to_numpy()
        self.biases = self.df["ethnicity"].astype(int).to_numpy()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        attr = torch.tensor([int(self.labels[idx]), int(self.biases[idx])], dtype=torch.long)
        return img, attr, path

transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        },
    "cifar10c": {
        "train": T.Compose([T.ToTensor(),]),
        "valid": T.Compose([T.ToTensor(),]),
        "test": T.Compose([T.ToTensor(),])
        },
    "waterbirds": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor(),])
        },
    "waterbirds_noise": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor(),])
        },
    "fairface": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor(),])
    },
}


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
    "cifar10c": {
        "train": T.Compose(
            [
                T.RandomCrop(32, padding=4),
                # T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "waterbirds": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4198, 0.4343, 0.3832],
                std=[0.2092, 0.2052, 0.2123],
            ),
        ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4198, 0.4343, 0.3832],
                std=[0.2092, 0.2052, 0.2123],
            ),
        ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4198, 0.4343, 0.3832],
                std=[0.2092, 0.2052, 0.2123],
            ),
        ]),
    },
    "waterbirds_noise": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4198, 0.4343, 0.3832],
                std=[0.2092, 0.2052, 0.2123],
            ),
        ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4198, 0.4343, 0.3832],
                std=[0.2092, 0.2052, 0.2123],
            ),
        ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4198, 0.4343, 0.3832],
                std=[0.2092, 0.2052, 0.2123],
            ),
        ]),
    },
    "fairface": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    },
}

transforms_preprcs_ae = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    },
    "cifar10c": {
        "train": T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "waterbirds": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=15, scale=(0.8, 1.2), translate=(0.1, 0.1)),
                T.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ]
        ),
        "valid": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ]
        ),
    },
    "waterbirds_noise": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=15, scale=(0.8, 1.2), translate=(0.1, 0.1)),
                T.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ]
        ),
        "valid": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ]
        ),
    },
    "fairface": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    },
}
def get_dataset(dataset, data_dir, dataset_split, transform_split, percent = 0.5, use_preprocess=None, image_path_list=None, use_type0=None, use_type1=None, balanced=False, extra_augs=False, image_size=224):
    dataset_category = dataset.split("-")[0]
    if use_preprocess:
        transform = transforms_preprcs[dataset_category][transform_split]
    else:
        transform = transforms[dataset_category][transform_split]
    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    # prepend extra train augmentations (debiasify-specific) if requested
    if transform_split == "train" and extra_augs:
        if hasattr(transform, "transforms"):
            base_transforms = list(transform.transforms)
        else:
            base_transforms = [transform]
        transform = T.Compose([T.RandomResizedCrop(image_size), T.RandomHorizontalFlip(), *base_transforms])
    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

    elif 'cifar10c' in dataset:
        # if use_type0:
        #     root = data_dir + f"/cifar10c_0805_type0/{percent}"
        # elif use_type1:
        #     root = data_dir + f"/cifar10c_0805_type1/{percent}"
        # else:
        root = data_dir + f"/cifar10c/{percent}"
        dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=use_type0, use_type1=use_type1)

    elif dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)
    
    elif dataset == "waterbirds":
        # root = data_dir + "/waterbirds"
        dataset = WaterbirdsDataset(root=data_dir, split=dataset_split, transform=transform, balanced=balanced)
    elif dataset == "waterbirds_noise":
        # root = data_dir + "/waterbirds_noise"
        dataset = WaterbirdsDataset(root=data_dir, split=dataset_split, transform=transform, balanced=balanced)
    elif dataset == "fairface":
        # root = data_dir + "/fairface"
        dataset = FairfaceDataset(root=data_dir, split=dataset_split, transform=transform)
    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset

