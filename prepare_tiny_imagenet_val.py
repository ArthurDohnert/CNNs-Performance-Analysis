import os
import shutil

print("Iniciando a reorganização do diretório de validação do Tiny ImageNet...")

# Caminho para o seu diretório de validação
val_dir = './data/tiny-imagenet-200/val'
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# 1. Ler o arquivo de anotações para mapear nome de imagem para classe
image_to_class = {}
with open(annotations_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        img_name, class_id = parts[0], parts[1]
        image_to_class[img_name] = class_id

print(f"Lidas {len(image_to_class)} anotações do arquivo {annotations_file}")

# 2. Criar os subdiretórios de classe
val_images_dir = os.path.join(val_dir, 'images')
for img_name, class_id in image_to_class.items():
    class_dir = os.path.join(val_dir, class_id)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

print("Subdiretórios de classe criados.")

# 3. Mover as imagens para seus respectivos diretórios de classe
for img_name, class_id in image_to_class.items():
    source_path = os.path.join(val_images_dir, img_name)
    dest_dir = os.path.join(val_dir, class_id)
    
    # Verifica se a imagem de origem existe antes de mover
    if os.path.exists(source_path):
        shutil.move(source_path, os.path.join(dest_dir, img_name))

print("Imagens de validação movidas para seus respectivos diretórios de classe.")

# 4. Limpeza: Remover o diretório 'images' vazio e o arquivo de anotações
try:
    if os.path.isdir(val_images_dir):
        os.rmdir(val_images_dir)
        print("Diretório 'images' original removido.")
    if os.path.exists(annotations_file):
        os.remove(annotations_file)
        print("Arquivo 'val_annotations.txt' removido.")
except OSError as e:
    print(f"Erro durante a limpeza (pode ser ignorado se os diretórios já estiverem vazios): {e}")

print("Reorganização concluída com sucesso!")
