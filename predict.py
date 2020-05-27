"""
Author: Pravesh Bawangade
File Name: predict.py

"""
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import torch
from skimage.color import lab2rgb, rgb2gray
from model import ColorizationUpsampling
import cv2


def predict_single():
    """
    Predict and show color image from input image
    :return: None
    """
    path = 'outputs/gray/img-8-epoch-29.jpg'
    img = Image.open(path)
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

    color_image_bgr = color_image.astype(np.float32)
    color_image_bgr = cv2.cvtColor(color_image_bgr, cv2.COLOR_RGB2BGR)
    color_image_bgr = cv2.resize(color_image_bgr, (380, 240))

    normalized_array = (color_image_bgr - np.min(color_image_bgr)) / (
            np.max(color_image_bgr) - np.min(color_image_bgr))  # this set the range from 0 till 1
    color_image_bgr = (normalized_array * 255).astype(np.uint8)
    gray = cv2.resize(gray, (380, 240))
    gray = np.stack((gray,) * 3, axis=-1)

    gray = (gray - np.min(gray)) / (
            np.max(gray) - np.min(gray))  # this set the range from 0 till 1
    gray = (gray * 255).astype(np.uint8)
    vis = np.concatenate((gray, color_image_bgr), axis=1)

    frame_normed = np.array(vis, np.uint8)

    cv2.imwrite(path[:-4]+"out.jpg", frame_normed)
    cv2.imshow("out", frame_normed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_video():
    """
    Predict and show real time image colorization.
    :return: None
    """
    cap = cv2.VideoCapture('input.avi')

    # For recording video
    frame_width = int(760)
    frame_height = int(240)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))

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
            color_image = color_image.transpose((1, 2, 0))
            color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
            color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
            color_image = lab2rgb(color_image.astype(np.float16))
            color_image_bgr = color_image.astype(np.float32)
            color_image_bgr = cv2.cvtColor(color_image_bgr, cv2.COLOR_RGB2BGR)
            color_image_bgr = cv2.resize(color_image_bgr, (380, 240))

            normalized_array = (color_image_bgr - np.min(color_image_bgr)) / (
                        np.max(color_image_bgr) - np.min(color_image_bgr))  # this set the range from 0 till 1
            color_image_bgr = (normalized_array * 255).astype(np.uint8)
            gray = cv2.resize(gray, (380,240))
            gray = np.stack((gray,)*3, axis=-1)

            gray = (gray - np.min(gray)) / (
                    np.max(gray) - np.min(gray))  # this set the range from 0 till 1
            gray = (gray * 255).astype(np.uint8)
            vis = np.concatenate((gray, color_image_bgr), axis=1)

            frame_normed = np.array(vis, np.uint8)

            cv2.imshow("image", frame_normed)
            out.write(frame_normed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_single()
    #predict_video()