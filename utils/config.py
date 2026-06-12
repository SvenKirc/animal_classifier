from torchvision import transforms

labels = {"butterfly": 0, "cat": 1, "chicken": 2, "cow": 3, "dog": 4, "elephant": 5, "horse": 6, "sheep": 7, "spider": 8, "squirrel": 9}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

epochs = 20
training = False