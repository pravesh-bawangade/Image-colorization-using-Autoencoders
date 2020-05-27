from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from model import ColorizationUpsampling
import cv2


def predict_single():
    img = Image.open('outputs/gray/img-0-epoch-18.jpg')
    img = img.resize((224,224))
    img_original = np.array(img)

    gray = rgb2gray(img_original)
    x = TF.to_tensor(gray).float()
    x.unsqueeze_(0)
    model = ColorizationUpsampling()
    model.load_state_dict(torch.load('checkpoints/model-epoch-22-losses-0.002910.pth',
                                     map_location=torch.device('cpu')))

    output = model(x)

    output = output.detach()
    color_image = torch.cat((x[0], output[0]), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float16))

    plt.imshow(color_image.astype('float'), interpolation='nearest')
    plt.show()


def predict_video():
    cv2_im = cv2.imread('outputs/gray/img-1-epoch-18.jpg')
    cap = cv2.VideoCapture(0)
    while True:
        ret, cv2_im = cap.read()
        if ret:
            cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_im)
            img = img.resize((224, 224))
            img_original = np.array(img)

            gray = rgb2gray(img_original)
            x = TF.to_tensor(gray).float()
            x.unsqueeze_(0)

            model = ColorizationUpsampling()
            model.load_state_dict(torch.load('checkpoints/model-epoch-22-losses-0.002910.pth',
                                             map_location=torch.device('cpu')))
            output = model(x)

            output = output.detach()
            color_image = torch.cat((x[0], output[0]), 0).numpy()
            color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
            color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
            color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
            color_image = lab2rgb(color_image.astype(np.float16))
            color_image_bgr = color_image.astype(np.float32)
            color_image_bgr = cv2.cvtColor(color_image_bgr, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", color_image_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_video()