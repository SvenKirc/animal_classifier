import torch
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import os

import config

labels = {"butterfly": 0, "cat": 1, "chicken": 2, "cow": 3, "dog": 4, "elephant": 5, "horse": 6, "sheep": 7, "spider": 8, "squirrel": 9}

class Pet_Data(Dataset):
    def __init__(self, data_directory):
        super().__init__()
        self.data_directory = data_directory
        self.indices = {}
        start_index = 0
        for folder_name in os.listdir(data_directory):
            label = labels[folder_name]
            end_index = len(os.listdir(data_directory + "/" + folder_name)) - 1
            self.indices[label] = [start_index, start_index + end_index]
            start_index = start_index + end_index + 1
        self.len = start_index
        self.transform = config.transform
    
    def __getitem__(self, index):
        # return the image, label at a certain index
        for label in self.indices:
            if index <= self.indices[label][1]:
                index -= self.indices[label][0]
                label = label
                break
        for animal in labels:
            if labels[animal] == label:
                index_animal = animal
                break
        image_path = self.data_directory + "/" + index_animal + "/" + os.listdir(self.data_directory + "/" + index_animal)[index]
        image = self.transform(Image.open(image_path).convert("RGB"))
        return image, label
    
    def __len__(self):
        return self.len

def get_dataloader():
    print("Processing Data")
    data_directory = "../dataset"
    pet_data = Pet_Data(data_directory)
    number_training_data = int(2/3 * len(pet_data))
    number_test_data = len(pet_data) - number_training_data
    train_data, test_data = random_split(pet_data, [number_training_data, number_test_data])
    train_dataloader = DataLoader(train_data, 64, True)
    test_dataloader = DataLoader(test_data, 64, False)
    return train_dataloader, test_dataloader

def main():
    train_dataloader, test_dataloader = get_dataloader()

if __name__ == "__main__":
    main()