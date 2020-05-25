# **Image Colorization using Autoencoders**
## Description:
    - In this project I tried to train autoencoder from scratch which can colorize grayscale images.
    - For encoder I used Resnet-18 Model [0-6]  and for decoder I used upsampling in pytorch.
    - This project is inspired from paper Colorful Image Colorization by Richard Zhang.
    
## Output:
   ![Output Image](outputs/gray/img-0-epoch-19.jpg) ![output image](outputs/color/img-0-epoch-19.jpg)
   ![Output Image](outputs/gray/img-1-epoch-19.jpg) ![output image](outputs/color/img-1-epoch-19.jpg)
   ![Output Image](outputs/gray/img-2-epoch-19.jpg) ![output image](outputs/color/img-2-epoch-19.jpg)
   ![Output Image](outputs/gray/img-3-epoch-29.jpg) ![output image](outputs/color/img-3-epoch-29.jpg)
   ![Output Image](outputs/gray/img-4-epoch-29.jpg) ![output image](outputs/color/img-4-epoch-29.jpg)
   ![Output Image](outputs/gray/img-5-epoch-29.jpg) ![output image](outputs/color/img-5-epoch-29.jpg)
   ![Output Image](outputs/gray/img-6-epoch-19.jpg) ![output image](outputs/color/img-6-epoch-19.jpg)
   ![Output Image](outputs/gray/img-7-epoch-29.jpg) ![output image](outputs/color/img-7-epoch-29.jpg)
   
## Train and Predict:
    - Use image_colorization.ipynb for training and prediction.
    
## Credits:
    - Richar Zhang paper on Colorful Colorization :
      -> https://arxiv.org/abs/1603.08511
        
    - Luke Melas :
      -> https://lukemelas.github.io/image-colorization.html