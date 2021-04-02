from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

def addGuass(image):
    im_arr = np.asarray(image)
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.06*255)
    new_array = aug.augment_image(im_arr)
    im = Image.fromarray(new_array)
    im.save("GaussAdded.tif")
    im.show()
    return im


def addSandP(image, spRatio):
    im_arr = np.asarray(image)
    aug = iaa.SaltAndPepper(p=spRatio)
    new_array = aug.augment_image(im_arr)
    im = Image.fromarray(new_array)
    im.save("saltAndPepperAdded.tif")
    im.show()
    return im
    
def main():
    image = Image.open('FileNameHere').convert('L')
    image.show()
    addSandP(image)
    addGuass(image)
    
