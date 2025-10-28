# src/utils/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(data_dir: str, batch_size: int, shuffle: bool, is_train: bool, num_workers: int = 4) -> DataLoader:
    """
    Cria um DataLoader aplicando data augmentation para treino e não para validação.

    Args:
        data_dir: Caminho para o diretório do dataset.
        batch_size: Tamanho do lote.
        shuffle: Se deve embaralhar os dados.
        is_train: Booleano que indica se é o conjunto de treino (para aplicar augmentation).
        num_workers: Número de processos para carregar os dados.
    """
    if is_train:
        # Pipeline de TREINO com Data Augmentation
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"INFO: Aplicando transformações de TREINO (com Data Augmentation) para: {data_dir}")
    else:
        # Pipeline de VALIDAÇÃO/TESTE (sem augmentation)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"INFO: Aplicando transformações de VALIDAÇÃO (sem Data Augmentation) para: {data_dir}")

    dataset = datasets.ImageFolder(data_dir, transform=preprocess)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

