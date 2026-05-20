from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
])