import numpy as np
import numbers
import math
import random
import scipy
import matplotlib
import PIL
from PIL import ImageFilter
import cv2
from PIL import ImageOps, Image


class RandomRotate(object):
    """
    Rotate video randomly by a random angle within given boundsi.
    Args:
        degrees (sequence or int): Range of degrees to randomly
        select from. If degrees is a number instead of sequence
        like (min, max), the range of degrees, will be
        (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees
        self.name = "RandomRotate"

    def __call__(self, clip):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        rotated = [img.rotate(angle) for img in clip]

        return rotated


class RandomResize(object):
    """
    Resize video by zooming in and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate

        self.interpolation = interp
        self.name = "RandomResize"

    def __call__(self, clip):
        scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

        im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip[0], np.ndarray):
            return [Image.fromarray(img).resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
            #return [scipy.misc.imresize(img, size=(new_h, new_w),interp=self.interpolation) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC


class RandomTranslate(object):
    """
      Shifting video in X and Y coordinates.
        Args:
            x (int) : Translate in x direction, selected
            randomly from [-x, +x] pixels.
            y (int) : Translate in y direction, selected
            randomly from [-y, +y] pixels.
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.name = "RandomTranslate"

    def __call__(self, clip):
        x_move = random.randint(-self.x, +self.x)
        y_move = random.randint(-self.y, +self.y)

        return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, x_move, 0, 1, y_move)) for img in clip]


class RandomShear(object):
    """
    Shearing video in X and Y directions.
    Args:
        x (int) : Shear in x direction, selected randomly from
        [-x, +x].
        y (int) : Shear in y direction, selected randomly from
        [-y, +y].
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.name = "RandomShear"

    def __call__(self, clip):
        x_shear = random.uniform(-self.x, self.x)
        y_shear = random.uniform(-self.y, self.y)

        return [img.transform(img.size, PIL.Image.AFFINE, (1, x_shear, 0, y_shear, 1, 0)) for img in clip]


class CenterCrop(object):
    """
    Extract center crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size
        self.name = "CenterCrop"

    def __call__(self, clip):
        crop_h, crop_w = self.size
        im_w, im_h = clip[0].size

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then ' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                                                                     h=crop_h))
            raise ValueError(error_msg)

        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))
        return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CornerCrop(object):
    """
    Extract corner crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size, crop_position=None):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

        if crop_position is None:
            self.randomize = True
        else:
            if crop_position not in ['c', 'tl', 'tr', 'bl', 'br']:
                raise ValueError("crop_position should be one of " +
                                 "['c', 'tl', 'tr', 'bl', 'br']")
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
        self.name = "CornerCrop"

    def __call__(self, clip):
        crop_h, crop_w = self.size
        im_w, im_h = clip[0].size

        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(0,len(self.crop_positions) - 1)]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((im_w - crop_w) / 2.))
            y1 = int(round((im_h - crop_h) / 2.))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_w
            y2 = crop_h
        elif self.crop_position == 'tr':
            x1 = im_w - crop_w
            y1 = 0
            x2 = im_w
            y2 = crop_h
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = im_h - crop_h
            x2 = crop_w
            y2 = im_h
        elif self.crop_position == 'br':
            x1 = im_w - crop_w
            y1 = im_h - crop_h
            x2 = im_w
            y2 = im_h

        return [img.crop((x1, y1, x2, y2)) for img in clip]


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size
        self.name = "RandomCrop"

    def __call__(self, clip):
        crop_h, crop_w = self.size
        im_w, im_h = clip[0].size

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h,
                                                                                     w=crop_w, h=crop_h))
            raise ValueError(error_msg)

        w1 = random.randint(0, im_w - crop_w)
        h1 = random.randint(0, im_h - crop_h)

        return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class HorizontalFlip(object):
    """
    Horizontally flip the video.
    """
    def __init__(self):
        self.name = "HorizontalFlip"

    def __call__(self, clip):
        return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]


class VerticalFlip(object):
    """
    Vertically flip the video.
    """
    def __init__(self):
        self.name = "VerticalFlip"

    def __call__(self, clip):

        return [img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in clip]


class GaussianBlur(object):
    """
    Augmenter to blur images using gaussian kernels.
    Args:
        sigma (float): Standard deviation of the gaussian kernel.
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self.name = "GaussianBlur"

    def __call__(self, clip):
        return [img.filter(ImageFilter.GaussianBlur(radius=self.sigma)) for img in clip]


class ElasticTransformation(object):
    """
    Augmenter to transform images by moving pixels locally around using
    displacement fields.
    See
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003
    for a detailed explanation.
    Args:
        alpha (float): Strength of the distortion field. Higher values mean
        more "movement" of pixels.
        sigma (float): Standard deviation of the gaussian kernel used to
        smooth the distortion fields.
        order (int): Interpolation order to use. Same meaning as in
        `scipy.ndimage.map_coordinates` and may take any integer value in
        the range 0 to 5, where orders close to 0 are faster.
        cval (int): The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.
        mode : Parameter that defines the handling of newly created pixels.
        May take the same values as in `scipy.ndimage.map_coordinates`,
        i.e. "constant", "nearest", "reflect" or "wrap".
    """
    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode
        self.name = "ElasticTransformation"

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        result = []
        nb_images = len(clip)
        for i in range(nb_images):
            image = clip[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            result.append(self._map_coordinates(
                clip[i],
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in result]
        else:
            return result

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2),"shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3),"image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result


class PiecewiseAffineTransform(object):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
     Args:
         displacement (init): gives distorted image depending on the value of displacement_magnification and displacement_kernel
         displacement_kernel (init): gives the blurry effect
         displacement_magnification (float): it magnifies the image
    """
    def __init__(self, displacement=0, displacement_kernel=0, displacement_magnification=0):
        self.displacement = displacement
        self.displacement_kernel = displacement_kernel
        self.displacement_magnification = displacement_magnification
        self.name = "PiecewiseAffineTransform"

    def __call__(self, clip):

        ret_img_group = clip
        im_size = clip[0].size
        image_w, image_h = im_size[0], im_size[1]

        displacement_map = np.random.rand(image_h, image_w, 2) * 2 * self.displacement - self.displacement
        displacement_map = cv2.GaussianBlur(displacement_map, None,
                                            self.displacement_kernel)
        displacement_map *= self.displacement_magnification * self.displacement_kernel
        displacement_map = np.floor(displacement_map).astype('int32')

        displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
        displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

        displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
        displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)

        return [PIL.Image.fromarray(np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(np.asarray(img).shape)) for img in clip]


class InvertColor(object):
    """
    Inverts the color of the video.
    """
    def __init__(self):
        self.name = "InvertColor"

    def __call__(self, clip):

        inverted = [ImageOps.invert(img) for img in clip]

        return inverted


class Add(object):
    """
    Add a value to all pixel intesities in an video.
    Args:
        value (int): The value to be added to pixel intesities.
    """

    def __init__(self, value=0):
        if value > 255 or value < -255:
            raise TypeError('The video is blacked or whitened out since ' +
                            'value > 255 or value < -255.')
        self.value = value
        self.name = "Add"

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.int32)
            image += self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Multiply(object):
    """
    Multiply all pixel intensities with given value.
    This augmenter can be used to make images lighter or darker.
    Args:
        value (float): The value with which to multiply the pixel intensities
        of video.
    """

    def __init__(self, value=1.0):
        if value < 0.0:
            raise TypeError('The video is blacked out since for value < 0.0')
        self.value = value
        self.name = "Multiply"

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.float32)
            image *= self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Pepper(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.
    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """
    def __init__(self, ratio=100):
        self.ratio = ratio
        self.name = "Pepper"

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float32)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.
    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """
    def __init__(self, ratio=100):
        self.ratio = ratio
        self.name = "Salt"

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float32)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class TemporalBeginCrop(object):
    """
    Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size
        self.name = "TemporalBeginCrop"

    def __call__(self, clip):
        out = clip[:self.size]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size
        self.name = "TemporalCenterCrop"

    def __call__(self, clip):
        center_index = len(clip) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size
        self.name = "TemporalRandomCrop"

    def __call__(self, clip):
        rand_end = max(0, len(clip) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(clip))
        print("begin index", begin_index)
        print("end index", end_index)

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class InverseOrder(object):
    """
    Inverts the order of clip frames.
    """
    def __init__(self):
        self.name = "InverseOrder"

    def __call__(self, clip):
        for i in range(len(clip)):
            nb_images = len(clip)
            return [clip[img] for img in reversed(range(0, nb_images))]


class Downsample(object):
    """
    Temporally downsample a video by deleting some of its frames.
    Args:
        ratio (float): Downsampling ratio in [0.0 <= ratio <= 1.0].
    """
    def __init__(self , ratio=1.0):
        if ratio < 0.0 or ratio > 1.0:
            raise TypeError('ratio should be in [0.0 <= ratio <= 1.0]. ' +
                            'Please use upsampling for ratio > 1.0')
        self.ratio = ratio
        self.name = "Downsample"

    def __call__(self, clip):
        nb_return_frame = np.floor(self.ratio * len(clip)).astype(int)
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i-1] for i in return_ind]


class Upsample(object):
    """
    Temporally upsampling a video by deleting some of its frames.
    Args:
        ratio (float): Upsampling ratio in [1.0 < ratio < infinity].
    """
    def __init__(self , ratio=1.0):
        if ratio < 1.0:
            raise TypeError('ratio should be 1.0 < ratio. ' +
                            'Please use downsampling for ratio <= 1.0')
        self.ratio = ratio
        self.name = "Upsample"

    def __call__(self, clip):
        nb_return_frame = np.floor(self.ratio * len(clip)).astype(int)
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i-1] for i in return_ind]


class TemporalFit(object):
    """
    Temporally fits a video to a given frame size by
    downsampling or upsampling.
    Args:
        size (int): Frame size to fit the video.
    """
    def __init__(self, size):
        if size < 0:
            raise TypeError('size should be positive')
        self.size = size
        self.name = "TemporalFit"

    def __call__(self, clip):
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=self.size)]

        return [clip[i-1] for i in return_ind]


class TemporalElasticTransformation(object):
    """
    Stretches or schrinks a video at the beginning, end or middle parts.
    In normal operation, augmenter stretches the beggining and end, schrinks
    the center.
    In inverse operation, augmenter shrinks the beggining and end, stretches
    the center.
    """
    def __init__(self, size):
        self.name = "TemporalElasticTransformation"

    def __call__(self, clip):
        nb_images = len(clip)
        new_indices = self._get_distorted_indices(nb_images)
        return [clip[i] for i in new_indices]

    def _get_distorted_indices(self, nb_images):
        inverse = random.randint(0, 1)

        if inverse:
            scale = random.random()
            scale *= 0.21
            scale += 0.6
        else:
            scale = random.random()
            scale *= 0.6
            scale += 0.8

        frames_per_clip = nb_images

        indices = np.linspace(-scale, scale, frames_per_clip).tolist()
        if inverse:
            values = [math.atanh(x) for x in indices]
        else:
            values = [math.tanh(x) for x in indices]

        values = [x / values[-1] for x in values]
        values = [int(round(((x + 1) / 2) * (frames_per_clip - 1), 0)) for x in values]
        return values


class Sequential(object):
    """
    Composes several augmentations together.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.rand = random_order

    def __call__(self, clip):
        if self.rand:
            rand_transforms = self.transforms[:]
            random.shuffle(rand_transforms)
            for t in rand_transforms:
                clip = t(clip)
        else:
            for t in self.transforms:
                clip = t(clip)

        return clip


class OneOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        select = random.choice(self.transforms)
        clip = select(clip)
        return clip


class SomeOf(object):
    """
    Selects a given number of augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations.
        N (int): The number of augmentations to select from the list.
        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, N, random_order=True):
        self.transforms = transforms
        self.rand = random_order
        if N > len(transforms):
            raise TypeError('The number of applied augmentors should be smaller than the given augmentation number')
        else:
            self.N = N

    def __call__(self, clip):
        if self.rand:
            tmp = self.transforms[:]
            selected_trans = [tmp.pop(random.randrange(len(tmp))) for _ in range(self.N)]
            # print("Selected transformations:", [trans.name for trans in selected_trans])
            for t in selected_trans:
                clip = t(clip)
            return clip
        else:
            indices = [i for i in range(len(self.transforms))]
            selected_indices = [indices.pop(random.randrange(len(indices)))
                                for _ in range(self.N)]
            selected_indices.sort()
            selected_trans = [self.transforms[i] for i in selected_indices]
            for t in selected_trans:
                clip = t(clip)
            return clip


class Sometimes(object):
    """
    Applies an augmentation with a given probability.
    Args:
        p (float): The probability to apply the augmentation.
        transform (an "Augmentor" object): The augmentation to apply.
    Example: Use this transform as follows:
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        sometimes(va.HorizontalFlip)
    """

    def __init__(self, p, transform):
        self.transform = transform
        if (p > 1.0) | (p < 0.0):
            raise TypeError('Expected p to be in [0.0 <= 1.0], ' +
                            'but got p = {0}'.format(p))
        else:
            self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            clip = self.transform(clip)
        return clip


class AffWildAugmentor1(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, video):
        # width, height = (np.min([img.size[0] for img in video]), np.min([img.size[1] for img in video]))
        seq = SomeOf([
            Add(value=100),
            # CenterCrop(size=(video[0].size[0] / 1.1, video[0].size[0] / 1.1)),
            # CornerCrop(size=(self.width / 1.1, self.height / 1.1)),
            ElasticTransformation(alpha=40, sigma=6, order=3, cval=100, mode="constant"),
            GaussianBlur(sigma=2.0),
            HorizontalFlip(),  #
            InvertColor(),
            Multiply(value=2),
            Pepper(ratio=80),
            PiecewiseAffineTransform(displacement=5, displacement_kernel=10, displacement_magnification=2),
            # RandomCrop(size=(int(min(height, width) / 1.05), int(min(width, height) / 1.05))),
            # RandomResize(rate=0.5),
            RandomRotate(degrees=20),
            RandomShear(x=0.2, y=0.2),  #
            RandomTranslate(x=50, y=50),
            Salt(),
            VerticalFlip(),
            # TemporalFit(),
            # TemporalElasticTransformation(),
            # TemporalBeginCrop(),
            # TemporalCenterCrop(),
            # TemporalRandomCrop(size=min(self.width, self.height)/1.1),
            # Upsample(ratio=2),
            # Downsample(),
            # InverseOrder()

        ], N=4, random_order=True)

        aug_vid = [img for img in seq(video)]
        # high_res_aug_vid = [img.resize((256, 256), Image.ANTIALIAS) for img in seq(video)]
        # aug_vid = np.array([np.asarray(i).astype('float32') / 255 for i in aug_vid])
        # high_res_aug_vid = np.array([np.asarray(i).astype('float32') / 255 for i in high_res_aug_vid])
        # print("Aug video shape:",aug_vid.shape)

        return aug_vid


class AffWildAugmentor2(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, video):
        # width, height = (np.min([img.size[0] for img in video]), np.min([img.size[1] for img in video]))
        seq = SomeOf([
            Add(value=100),
            # CenterCrop(size=(video[0].size[0] / 1.1, video[0].size[0] / 1.1)),
            # CornerCrop(size=(self.width / 1.1, self.height / 1.1)),
            ElasticTransformation(alpha=40, sigma=6, order=3, cval=100, mode="constant"),
            GaussianBlur(sigma=2.0),
            HorizontalFlip(),  #
            InvertColor(),
            Multiply(value=2),
            Pepper(ratio=80),
            PiecewiseAffineTransform(displacement=5, displacement_kernel=10, displacement_magnification=2),
            # RandomCrop(size=(int(min(height, width) / 1.05), int(min(width, height) / 1.05))),
            # RandomResize(rate=0.5),
            RandomRotate(degrees=20),
            RandomShear(x=0.2, y=0.2),  #
            RandomTranslate(x=50, y=50),
            Salt(),
            VerticalFlip(),
            # TemporalFit(),
            # TemporalElasticTransformation(),
            # TemporalBeginCrop(),
            # TemporalCenterCrop(),
            # TemporalRandomCrop(size=min(self.width, self.height)/1.1),
            # Upsample(ratio=2),
            # Downsample(),
            # InverseOrder()

        ], N=2, random_order=True)

        aug_vid = [img for img in seq(video)]
        # high_res_aug_vid = [img.resize((256, 256), Image.ANTIALIAS) for img in seq(video)]
        # aug_vid = np.array([np.asarray(i).astype('float32') / 255 for i in aug_vid])
        # high_res_aug_vid = np.array([np.asarray(i).astype('float32') / 255 for i in high_res_aug_vid])
        # print("Aug video shape:",aug_vid.shape)

        return aug_vid


class AffWildAugmentor3(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, video):
        # width, height = (np.min([img.size[0] for img in video]), np.min([img.size[1] for img in video]))
        seq = SomeOf([
            # Add(value=100),
            # CenterCrop(size=(video[0].size[0] / 1.1, video[0].size[0] / 1.1)),
            # CornerCrop(size=(self.width / 1.1, self.height / 1.1)),
            # ElasticTransformation(alpha=40, sigma=6, order=3, cval=100, mode="constant"),
            # GaussianBlur(sigma=2.0),
            # HorizontalFlip(),  #
            # InvertColor(),
            # Multiply(value=2),
            Pepper(ratio=80),
            # PiecewiseAffineTransform(displacement=5, displacement_kernel=10, displacement_magnification=2),
            # RandomCrop(size=(int(min(height, width) / 1.05), int(min(width, height) / 1.05))),
            # RandomResize(rate=0.5),
            # RandomRotate(degrees=20),
            # RandomShear(x=0.2, y=0.2),  #
            # RandomTranslate(x=50, y=50),
            Salt(),
            # VerticalFlip(),
            # TemporalFit(),
            # TemporalElasticTransformation(),
            # TemporalBeginCrop(),
            # TemporalCenterCrop(),
            # TemporalRandomCrop(size=min(self.width, self.height)/1.1),
            # Upsample(ratio=2),
            # Downsample(),
            # InverseOrder()

        ], N=1, random_order=True)

        aug_vid = [img for img in seq(video)]
        # high_res_aug_vid = [img.resize((256, 256), Image.ANTIALIAS) for img in seq(video)]
        # aug_vid = np.array([np.asarray(i).astype('float32') / 255 for i in aug_vid])
        # high_res_aug_vid = np.array([np.asarray(i).astype('float32') / 255 for i in high_res_aug_vid])
        # print("Aug video shape:",aug_vid.shape)

        return aug_vid

