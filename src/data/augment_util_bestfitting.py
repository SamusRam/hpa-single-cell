from src.commons.config.config_bestfitting import *
from albumentations import ShiftScaleRotate, Compose, VerticalFlip, HorizontalFlip, RandomBrightness
import cv2


def augment_rot(image):
    return rotate_image(image, np.random.randint(-90, 90))


def augment_rot_vert_flip(image):
    image = rotate_image(image, np.random.randint(-90, 90))
    image = np.flipud(image)
    return image


def augment_rot_hor_flip(image):
    image = rotate_image(image, np.random.randint(-90, 90))
    image = np.fliplr(image)
    return image

def augment_rot_hor_flip(image):
    image = rotate_image(image, np.random.randint(-90, 90))
    image = np.fliplr(image)
    return image


scale_shift = ShiftScaleRotate(border_mode=0, scale_limit=0.4, rotate_limit=0, p=0.25)
def augment_shift_scale(image):
    aug_output = scale_shift(image=image)
    return aug_output['image']



def train_multi_augment2(image):
    augment_func_list = [
        lambda image: (image), # default
        augment_flipud,                    # up-down
        augment_fliplr,                    # left-right
        augment_transpose,                 # transpose
        augment_flipud_lr,                 # up-down left-right
        augment_flipud_transpose,          # up-down transpose
        augment_fliplr_transpose,          # left-right transpose
        augment_flipud_lr_transpose       # up-down left-right transpose
    ]
    c = np.random.choice(len(augment_func_list))
    image = augment_func_list[c](image)
    return image


brightness_aug = RandomBrightness(limit=(-0.05, 0.3), p=0.25)

def train_multi_augment3(image):
    augment_func_list = [
        lambda image: (image), # default
        augment_flipud,                    # up-down
        augment_fliplr,                    # left-right
        augment_transpose,                 # transpose
        augment_flipud_lr,                 # up-down left-right
        augment_flipud_transpose,          # up-down transpose
        augment_fliplr_transpose,          # left-right transpose
        augment_flipud_lr_transpose,       # up-down left-right transpose
    ]
    c = np.random.choice(len(augment_func_list))
    image = augment_func_list[c](image)
    image = augment_shift_scale(image)
    image = brightness_aug(image=image)['image']
    return image


# source: https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides/47248339#47248339
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


vert_hor_flip = Compose([VerticalFlip(always_apply=True),
                         HorizontalFlip(always_apply=True)])



scale_shift = ShiftScaleRotate(always_apply=True, border_mode=0, scale_limit=0.4, rotate_limit=0)


def augment_default(image, mask=None):
    if mask is None:
        return image
    else:
        return image, mask

def augment_flipud(image, mask=None):
    image = np.flipud(image)
    if mask is None:
        return image
    else:
        mask = np.flipud(mask)
        return image, mask

def augment_fliplr(image, mask=None):
    image = np.fliplr(image)
    if mask is None:
        return image
    else:
        mask = np.fliplr(mask)
        return image, mask

def augment_transpose(image, mask=None):
    image = np.transpose(image, (1, 0, 2))
    if mask is None:
        return image
    else:
        if len(mask.shape) == 2:
            mask = np.transpose(mask, (1, 0))
        else:
            mask = np.transpose(mask, (1, 0, 2))
        return image, mask

def augment_flipud_lr(image, mask=None):
    image = np.flipud(image)
    image = np.fliplr(image)
    if mask is None:
        return image
    else:
        mask = np.flipud(mask)
        mask = np.fliplr(mask)
        return image, mask

def augment_flipud_transpose(image, mask=None):
    if mask is None:
        image = augment_flipud(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_flipud(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask

def augment_fliplr_transpose(image, mask=None):
    if mask is None:
        image = augment_fliplr(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_fliplr(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask

def augment_flipud_lr_transpose(image, mask=None):
    if mask is None:
        image = augment_flipud(image, mask=mask)
        image = augment_fliplr(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_flipud(image, mask=mask)
        image, mask = augment_fliplr(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask
