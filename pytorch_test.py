from PIL import Image
import torch
from torchvision import transforms


transform = transforms.Compose([
    transforms.RandomCrop(300),
    transforms.ToTensor(),
])

im_filepath = 'pexels-oswald-elsaboath-7061955_720x480.jpg'


im_pil = Image.open(im_filepath)
im_pth = transform(im_pil)
im_pth2 = transform(im_pil)


images = [im_pth, im_pth2]
batch = torch.stack(images)


print(batch.shape)
