from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch import save, load, no_grad, max
from matplotlib import pyplot as plt


class Model:
    def __init__(self, model, state_path):
        self.model = model
        self.state_path = state_path

    def train(self, train_set, batch_size, epochs):
        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        loss_function = CrossEntropyLoss()

        # network training:
        train_loss_values = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['ecg']
                targets = data['label']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 20 == 19:  # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, epoch_loss / 20))
                    epoch_loss = 0.0

            # Save loss value for current epoch run
            train_loss_values.append(running_loss / len(train_set))

        print('Finished Training')
        plt.title(f'Model Loss for {epochs} epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(train_loss_values)
        plt.show()

        # save net state dict
        #PATH = './speech_net.pth'
        save(self.model.state_dict(), self.state_path)

    def test(self, test_set, batch_size):
        data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        self.model.load_state_dict(load(self.state_path))
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with no_grad():
            for data in data_loader:
                inputs = data['ecg']
                targets = data['label']

                outputs = self.model(inputs)
                _, predicted = max(outputs.data, 1)

                c = (predicted == targets).squeeze()
                for i in range(4):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                total += targets.size(0)
                correct += (predicted == targets).sum().item()