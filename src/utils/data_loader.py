# src/utils/data_loader.py


###
### Functions that load the data
###


# imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


# implementations

def get_dataloader(data_dir: str, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Cria e retorna um DataLoader para um dataset de imagens organizado em pastas por classe.

    Esta função é ideal para datasets como ImageNet, Tiny ImageNet, ou qualquer
    outro que siga a estrutura:
        data_dir/
        ├── class_a/
        │   ├── xxx.png
        │   ├── xxy.png
        │   └── ...
        └── class_b/
            ├── zzz.png
            └── ...

    Args:
        data_dir: Caminho para o diretório raiz do dataset (ex: '.../train' ou '.../val').
        batch_size: O tamanho do lote para o DataLoader.
        shuffle: Se deve embaralhar os dados (True para treino, False para validação/teste).

    Returns:
        Um objeto torch.utils.data.DataLoader configurado.
    """
    # Define as transformações padrão para modelos pré-treinados em ImageNet
    # 1. Redimensiona para 256x256
    # 2. Corta o centro para 224x224 (tamanho de entrada comum para muitos modelos)
    # 3. Converte para tensor
    # 4. Normaliza com as médias e desvios padrão do ImageNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Cria o dataset usando ImageFolder, que automaticamente encontra as classes e imagens
    dataset = datasets.ImageFolder(data_dir, transform=preprocess)

    # Cria o DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # Use múltiplos processos para carregar os dados mais rápido
        pin_memory=True # Otimização para transferência de dados para a GPU
    )

    return dataloader


def load_preprocessed(path, batch_size=64):
    """
    Carrega tensores pré-processados e salvos em um único arquivo.
    (Mantido para compatibilidade, caso seja usado em outro lugar).
    """
    data = torch.load(path)
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    num_classes = data["num_classes"]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes
