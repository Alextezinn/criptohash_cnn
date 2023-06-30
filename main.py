from pathlib import Path
import pickle

import psycopg2
import torch
import torchvision.transforms as transforms
from PIL import Image

from database import Database
from model import ImageEncoder


model = ImageEncoder()
model = torch.load('model.pth')
model.eval()


def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image_transform = transform(image)

    return torch.Tensor(image_transform).unsqueeze(0)


def insert_data_table():
    for path in Path('./img').iterdir():
        image = Image.open(path).convert('RGB')
        preprocessed = preprocess(image)
        output_image = model(preprocessed)
        list_output_image = output_image[0].tolist()
        serialization_obj = pickle.dumps(list_output_image)
        Database.insert_featuremap(psycopg2.Binary(serialization_obj), Path.cwd() / path)

def main():
    records = Database.select_all()

    for record in records:
        featuremap, path_image = record
        featuremap = torch.tensor(pickle.loads(featuremap))

        image = Image.open("uKF9WyvZlbE.jpg").convert('RGB')
        preprocessed = preprocess(image)
        output_image = model(preprocessed)

        cos = torch.nn.CosineSimilarity(dim=1)
        cos = cos(torch.Tensor(featuremap.unsqueeze(0)), torch.Tensor(output_image))

        if cos[0] > 0.95:
            print(path_image)


if __name__ == "__main__":
    # раскоментить и выполнить 1 раз
    # Database.create_table()
    # insert_data_table()
    main()
