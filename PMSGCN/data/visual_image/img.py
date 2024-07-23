import numpy as np
import cv2

import torch

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) #从(394,394,3)变成（512,512,3）  区域插值


def get_square_bbox(bbox):
    """Makes square bbox from any bbox by stretching of minimal length side

    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox #264,264,658,658
    width, height = right - left, lower - upper #394,394

    x_center, y_center = (right + left) // 2, (lower + upper) // 2 #461,461
    new_width, new_height = int(scale * width), int(scale * height) # 1*

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def image_batch_to_numpy(image_batch):
    image_batch = to_numpy(image_batch) #tensor(4,3,384,384)->nd
    image_batch = np.transpose(image_batch, (0, 2, 3, 1)) # BxCxHxW -> BxHxWxC   4,384,384,3
    return image_batch


def image_batch_to_torch(image_batch):
    image_batch = np.transpose(image_batch, (0, 3, 1, 2)) # BxHxWxC -> BxCxHxW  4,3,384,384
    image_batch = to_torch(image_batch).float()  #tensor
    return image_batch


def normalize_image(image):
    """Normalizes image using ImageNet mean and std

    Args:
        image numpy array of shape (h, w, 3): image

    Returns normalized_image numpy array of shape (h, w, 3): normalized image
    """
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)  #IMAGENET_STD:[0.229 0.224 0.225]    IMAGENET_MEAN:[0.485 0.456 0.406]
     #clip:  <0 -->0   >255-->255