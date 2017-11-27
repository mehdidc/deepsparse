import torch

import torchvision.transforms as transforms
import torchvision.datasets as dset


def load_dataset(dataset_name, split='full'):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root='/home/mcherti/work/data/mnist', 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        return dataset
    elif dataset_name == 'cifar':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/cifar10/img_classes',
            transform=transforms.Compose([
            transforms.ToTensor(),
         ]))
        return dataset
 
    elif dataset_name == 'coco':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/coco',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'quickdraw':
        X = (np.load('/home/mcherti/work/data/quickdraw/teapot.npy'))
        X = X.reshape((X.shape[0], 28, 28))
        X  = X / 255.
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
        dataset = TensorDataset(X, X)
        return dataset
    elif dataset_name == 'shoes':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/shoes/ut-zap50k-images/Shoes',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'footwear':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/shoes/ut-zap50k-images',
            transform=transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'celeba':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/celeba',
            transform=transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'birds':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/birds/'+split,
            transform=transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'sketchy':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/sketchy/'+split,
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            Gray()
         ]))
        return dataset
 
    elif dataset_name == 'fonts':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/fonts/'+split,
            transform=transforms.Compose([
            transforms.ToTensor(),
            Invert(),
            Gray(),
         ]))
        return dataset
    else:
        raise ValueError('Error : unknown dataset')
