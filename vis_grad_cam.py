import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19

from gradcam import GradCam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Grad-CAM')
    parser.add_argument('--image_name', default='both.png', type=str, help='the tested image name')
    parser.add_argument('--save_name', default='grad_cam.png', type=str, help='saved image name')

    opt = parser.parse_args()

    IMAGE_NAME = opt.image_name
    SAVE_NAME = opt.save_name
    test_image = (transforms.ToTensor()(Image.open(IMAGE_NAME))).unsqueeze(dim=0)
    model = vgg19(pretrained=True)
    if torch.cuda.is_available():
        test_image = test_image.cuda()
        model.cuda()
    grad_cam = GradCam(model)
    feature_image = grad_cam(test_image).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save(SAVE_NAME)
