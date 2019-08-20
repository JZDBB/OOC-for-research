import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, CIFAR10
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# RGB
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# gray image
# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
#     ])


dataset = CIFAR10('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=2),  # b, 16, 14, 14
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),  # b, 16, 13, 13
            nn.Conv2d(64, 256, 5, stride=1, padding=2),  # b, 32, 7, 7
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),  # b, 32, 6, 6
            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # b, 64, 3, 3
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # b, 64, 3, 3
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # b, 64, 3, 3
            nn.ReLU(True),
            # b, 256, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1),  # b, 512, 3, 3
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
f = open("print.txt", "w+")
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    msg = 'epoch [{}/{}], loss:{:.4f}\n'.format(epoch+1, num_epochs, loss.item())
    f.write(msg)
    print(msg)

    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
f.close()