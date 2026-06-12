import torch
from torch import nn

from utils import data_handler, model, config

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train(classifier):
    train_dataloader, test_dataloader = data_handler.get_dataloader()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=0.01)

    classifier.training

    training_loss = 0

    for epoch in range(config.epochs):
        classifier.train()
        training_loss = 0
        for X_train, y_train in train_dataloader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred = classifier(X_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(test_dataloader)

        classifier.eval()
        with torch.inference_mode():
            test_loss = 0
            for X_test, y_test in test_dataloader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                y_test_pred = classifier(X_test)
                loss = loss_fn(y_test_pred, y_test)
                test_loss += loss.item()
            test_loss /= len(test_dataloader)
        print(f"Epoch: {epoch} | Train Loss: {training_loss:.4f} | Test Loss: {test_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(classifier.state_dict(), "animal_classifier.pth")
            print("Saved the models state dict.")


def main():
    animal_classifier = model.Animal_Classifier()
    animal_classifier = animal_classifier.to(device)
    if config.training == True:
        train(animal_classifier)
    
if __name__ == "__main__":
    main()