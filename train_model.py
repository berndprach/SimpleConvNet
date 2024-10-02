import torch
import torchvision
from torch.optim import SGD, lr_scheduler
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms as tfs

from simple_conv_net import get_conv_net


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.1


def get_cifar_10():
    cifar_channel_means = (0.4914, 0.4822, 0.4465)
    center_data = tfs.Normalize(cifar_channel_means, (1., 1., 1.))
    train_val_ds = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        transform=tfs.Compose([tfs.ToTensor(), center_data]),
        download=True
    )
    train_ds, val_ds = random_split(train_val_ds, [45000, 5000])

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    return train_dl, val_dl


def get_augmentation(h=32, w=32, crop_size=4):
    crop = tfs.RandomCrop((h, w), padding=crop_size, padding_mode="reflect")
    flip = tfs.RandomHorizontalFlip()
    erase = tfs.RandomErasing(p=1., scale=(1 / 16, 1 / 16), ratio=(1., 1.))
    return tfs.Compose([crop, flip, erase])


class CrossEntropyWithTemperature:
    def __init__(self, temperature=1., **kwargs):
        self.temperature = temperature
        self.std_xent = torch.nn.CrossEntropyLoss(**kwargs)

    def __call__(self, score_batch, label_batch):
        score_batch /= self.temperature
        return self.std_xent(score_batch, label_batch) * self.temperature


def train_model():
    train_dl, val_dl = get_cifar_10()
    model = get_conv_net()
    model.to(DEVICE)

    loss_function = CrossEntropyWithTemperature(temperature=8)
    optimizer = SGD(model.parameters(), lr=0., momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.OneCycleLR(sgd, max_lr=LR, total_steps=24)
    augment = get_augmentation()

    for epoch_nr in range(24):
        epoch_losses = []
        epoch_accs = []

        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch = augment(x_batch)

            predictions = model(x_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.detach())
            batch_accuracies = torch.eq(predictions.argmax(dim=1), y_batch)
            epoch_accs.append(batch_accuracies.detach())

        scheduler.step()
        epoch_loss = torch.stack(epoch_losses).mean()
        epoch_acc = torch.cat(epoch_accs).float().mean()
        print(f"Epoch {epoch_nr + 1}: "
              f"Loss: {epoch_loss:.3f}, "
              f"Acc: {epoch_acc:.1%}")

    val_losses = []
    val_accs = []
    for x_batch, y_batch in val_dl:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        predictions = model(x_batch)
        loss = loss_function(predictions, y_batch)
        val_losses.append(loss.detach())

        batch_accuracies = torch.eq(predictions.argmax(dim=1), y_batch)
        val_accs.append(batch_accuracies.detach())

    epoch_loss = torch.stack(val_losses).mean()
    epoch_accs = torch.cat(val_accs).float().mean()
    print(f"Val Loss: {epoch_loss:.3f}, Val Acc: {epoch_accs:.1%}")

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()
