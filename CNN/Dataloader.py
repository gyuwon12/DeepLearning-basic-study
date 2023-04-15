import torch
from torch.utils.data.dataloader import DataLoader

def make_dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
    
    return train_loader, test_loader