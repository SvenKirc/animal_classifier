import torch
from torch import nn

from tqdm import tqdm

from utils import data_handler, model, config

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def train(classifier):
    train_dataloader, test_dataloader = data_handler.get_dataloader("dataset")

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=0.01)

    training_loss = 0

    for epoch in range(config.epochs):
        classifier.train()
        training_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        training_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for X_train, y_train in training_bar:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred = classifier(X_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            _, predicted_class = y_pred.max(1)
            for i_prediction in range(len(predicted_class)):
                total_predictions += 1
                if predicted_class[i_prediction] == y_train[i_prediction]:
                    correct_predictions += 1
            accuracy = correct_predictions / total_predictions
            average_loss = training_loss / (training_bar.n + 1)
            training_bar.set_postfix({
                "loss": f"{average_loss:.4f}",
                "acc": f"{accuracy:.4f}"
                })

        classifier.eval()
        with torch.inference_mode():
            test_loss = 0
            testing_bar = tqdm(test_dataloader, desc=f"Testing")
            total_predictions = 0
            correct_predictions = 0
            average_loss = 0
            for X_test, y_test in testing_bar:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                y_test_pred = classifier(X_test)
                _, predicted_class = y_test_pred.max(1)
                loss = loss_fn(y_test_pred, y_test)
                test_loss += loss.item()
                for i_prediction in range(len(predicted_class)):
                    total_predictions += 1
                    if predicted_class[i_prediction] == y_test[i_prediction]:
                        correct_predictions += 1
                accuracy = correct_predictions / total_predictions
                average_loss = test_loss / (testing_bar.n + 1)
                testing_bar.set_postfix({
                            "loss": f"{average_loss:.4f}",
                            "acc": f"{accuracy:.4f}"
                            })                
            test_loss /= len(test_dataloader)
        
        if epoch % 10 == 0 and epoch > 0:
            torch.save(classifier.state_dict(), "animal_classifier.pth")

def main():
    animal_classifier = model.Animal_Classifier()
    animal_classifier = animal_classifier.to(device)
    if config.training == True:
        train(animal_classifier)
    
if __name__ == "__main__":
    main()