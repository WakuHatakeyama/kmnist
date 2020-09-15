import torch
import os

def train(train_loader, val_loader, model, criterion, optimizer, n_epochs=50, save_weight=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(n_epochs):
        running_loss = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%50 == 49:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch+1, i+1, running_loss/50))
                running_loss = 0.0

        correct = 0
        total = 0
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Val accuracy: {round(100*float(correct/total), 2)} %, loss: {round(val_loss/i, 3)}')

    if save_weight:
        os.makedirs('output/', exist_ok=True)
        torch.save(model.state_dict(), f'output/{round(100*float(correct/total), 2)}.pth')
