import torch
from torchvision import datasets, transforms
from matplotlib import pyplot
from os import cpu_count

convolution_kernel_size = 3
pooling_kernel_size = 2
dropout_percent = 0.5
image_size = 256


class Block(torch.nn.Module):
    def __init__(self, n_input_channels, n_output_channels):
        super().__init__()
        self.convolution = torch.nn.Conv2d(n_input_channels, n_output_channels,
                                           convolution_kernel_size, padding='same')

        self.activation = torch.nn.ReLU()

        self.batchnorm = torch.nn.BatchNorm2d(n_output_channels)

        self.pooling = torch.nn.MaxPool2d(pooling_kernel_size)

    def forward(self, x):
        return self.pooling(self.batchnorm(self.activation(self.convolution(x))))


class CNN(torch.nn.Module):
    def __init__(self):
        global image_size
        super().__init__()
        sizes = [(3, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
        self.final_size = sizes[-1][1]
        self.blocks = torch.nn.Sequential(*[Block(in_, out_) for in_, out_ in sizes])

        final_image_size = image_size // 2 ** len(sizes)

        self.average_pool = torch.nn.AvgPool2d(final_image_size)
        self.dropout = torch.nn.Dropout(dropout_percent)
        self.activation = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(self.final_size, 1000)
        self.linear2 = torch.nn.Linear(1000, 2)

    def forward(self, x):
        x = self.blocks(x)
        x = self.dropout(self.average_pool(x).view((-1, self.final_size)))
        x = self.dropout(self.activation(self.linear1(x)))
        return self.linear2(x)


def get_accuracy():
    global model, val_len, val_loader
    correct_ = 0
    model.eval()
    with torch.no_grad():
        for x_batch_, y_batch_ in val_loader:
            logits_ = model(x_batch_)

            probabilities_ = torch.softmax(logits_, dim=1)
            _, predicted_ = torch.max(probabilities_, 1)

            correct_ += (predicted_ == y_batch_).sum().item()
    return correct_ / val_len


if __name__ == '__main__':
    batch_size = 32

    data_folder_path = r"C:\Users\tbaka\Jupyter\Data Science Projects\medical_detection\chest_xray\train"
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.RandomResizedCrop((image_size, image_size), (0.8, 1.0)),
                                    transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(data_folder_path, transform=transform)

    data_folder_path = r"C:\Users\tbaka\Jupyter\Data Science Projects\medical_detection\chest_xray\val"
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    val_dataset = datasets.ImageFolder(data_folder_path, transform=transform)

    # for parallel processing
    n_cores = cpu_count() - 1

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=n_cores)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=n_cores)

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    model = CNN()
    print('initial accuracy:', get_accuracy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    record_loss = []

    # Training the model
    max_epochs = 30
    train_accuracy = [0.0] * max_epochs
    val_accuracy = [0.0] * max_epochs
    n_batches = len(train_loader)
    best_acc = 0.0

    for i in range(max_epochs):
        correct = 0
        model.train()
        k = 0
        for x_batch, y_batch in train_loader:
            k += 1
            print('\repoch:', i + 1, 'batch:', k, 'out of', n_batches, end='')
            # forward pass
            logits = model(x_batch)

            # backward_pass
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(logits, y_batch)
            loss.backward()

            # update parameters
            optimizer.step()

            with torch.no_grad():
                probabilities = torch.softmax(logits, dim=-1)
                _, predicted = torch.max(probabilities, 1)
                correct += (predicted == y_batch).sum().item()

        train_accuracy[i] = correct / train_len
        val_accuracy[i] = get_accuracy()
        if val_accuracy[i] > best_acc:
            # save parameters with the best validation accuracy
            best_acc = val_accuracy[i]
            torch.save(model.state_dict(), "model_weights.pth")

        print('\nepoch:', i + 1, 'train accuracy:', train_accuracy[i], 'val accuracy:', val_accuracy[i])

    del(train_loader, val_loader)
    print(f"Accuracy on train set: {train_accuracy[-1]:.2f}%")
    print(f"Accuracy on val set: {val_accuracy[-1]:.2f}%")

    dummy = range(1, max_epochs + 1)
    pyplot.plot(dummy, train_accuracy, 'o-')
    pyplot.plot(dummy, val_accuracy, 'o-')
    pyplot.xlabel('epoch')
    pyplot.ylabel('estimated accuracy')
    pyplot.legend(['Train', 'Val'])
    pyplot.savefig("accuracy_logging.png")

    # Save optimizer
    torch.save(optimizer.state_dict(), "optimizer.pth")
