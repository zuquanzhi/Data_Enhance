import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3*64*64),  # 生成64x64的图像，RGB
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 3, 64, 64)  # reshape为图像形状
        return out

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3*64*64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img = img.view(img.size(0), -1)  # 将图像展平
        out = self.fc(img)
        return out

# 数据集类（这里使用自己的数据集生成运动模糊图像）
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img

# 初始化网络、损失函数和优化器
def initialize_network():
    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return generator, discriminator, criterion, optim_g, optim_d

# 训练GAN
def train_gan(generator, discriminator, dataloader, criterion, optim_g, optim_d, epochs=10):
    for epoch in range(epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # 训练判别器
            optim_d.zero_grad()
            
            # 真实图像判别
            real_images = real_images.cuda()
            output = discriminator(real_images)
            d_loss_real = criterion(output, real_labels)
            d_loss_real.backward()

            # 生成的假图像判别
            z = torch.randn(batch_size, 100).cuda()
            fake_images = generator(z)
            output = discriminator(fake_images.detach())
            d_loss_fake = criterion(output, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            optim_d.step()

            # 训练生成器
            optim_g.zero_grad()
            output = discriminator(fake_images)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optim_g.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

# 使用 PyTorch 数据加载
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageDataset('/home/test/Desktop/Data_Enhance/output', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化和训练
generator, discriminator, criterion, optim_g, optim_d = initialize_network()
train_gan(generator, discriminator, dataloader, criterion, optim_g, optim_d, epochs=10)
