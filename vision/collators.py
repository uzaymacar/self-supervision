"""Data collators for self-supervised pretext tasks in vision."""
import random
import itertools
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import hamming

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import VOCSegmentation
from torchvision.transforms import transforms


class DataCollator(object):
    def __init__(self):
        pass

    def __call__(self, examples):
        pass

    def _preprocess_batch(self, examples):
        """
        Preprocesses tensors: returns the current examples as batch for supervised
        classification datasets.
        """
        # TODO: What if the data is passed in the desired x, y form already?!
        examples = [(example[0], torch.tensor(example[1]).long()) for example in examples]
        if all(x.shape == examples[0][0].shape for x, y in examples):
            x, y = tuple(map(torch.stack, zip(*examples)))
            return x, y
        else:
            raise ValueError('Examples must contain the same shape!')


class RotationCollator(DataCollator):
    """
    Data collator used for rotation classification task.

    :param (int) num_rotations: number of classes of rotations to use
    :param (str) rotation_procedure: set to 'all' for including all possible `num_rotations`
           rotations for each example in batch, or set to 'random' to randomly choose rotations
           for each example in batch
    """
    def __init__(self, num_rotations=4, rotation_procedure='all'):
        super(RotationCollator).__init__()
        self.num_rotations = num_rotations
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
        # We are excluding the last item, 360 degrees, since it is equivalent to 0 degrees

        assert rotation_procedure in ['all', 'random']
        self.rotation_procedure = rotation_procedure

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, y = batch
        batch_size = x.shape[0]

        if self.rotation_procedure == 'random':
            for i in range(batch_size):
                theta = np.random.choice(self.rotation_degrees, size=1)[0]
                x[i] = self.rotate_image(x[i].unsqueeze(0), theta=theta).squeeze(0)
                y[i] = torch.tensor(self.rotation_degrees.index(theta)).long()

        elif self.rotation_procedure == 'all':
            x_, y_ = [], []
            for theta in self.rotation_degrees:
                x_.append(self.rotate_image(x.clone(), theta=theta))
                y_.append(torch.tensor(self.rotation_degrees.index(theta)).long().repeat(batch_size))

            x, y = torch.cat(x_), torch.cat(y_)

            # Shuffle images and labels to get rid of the fixed [0, 90, 180, 270] order
            # NOTE: Effective batch size is `batch_size` * 4
            permutation = torch.randperm(batch_size * 4)
            x, y = x[permutation], y[permutation]

        return x, y

    @staticmethod
    def get_rotation_matrix(theta, mode='degrees'):
        """
        Computes and returns the rotation matrix.

        :param (int or float) theta: integer angle value for `mode`='degrees' and float angle value
               for `mode`='radians'
        :param (str) mode: set to 'degrees' or 'radians'
        :return: rotation matrix
        """
        assert mode in ['degrees', 'radians']

        if mode == 'degrees':
            theta *= np.pi / 180

        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def rotate_image(self, x, theta):
        """
        Rotates tensors in batch. Implemented according to:
        https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports

        :param (torch.Tensor) x: tensor image(s) to be rotated
        :param (int or float) theta: integer angle value for `mode`='degrees' and float angle value
               for `mode`='radians'
        """

        dtype = x.dtype
        rotation_matrix = self.get_rotation_matrix(theta=theta, mode='degrees')[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rotation_matrix, x.shape, align_corners=True).type(dtype)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


class CountingCollator(DataCollator):
    """
    Data collator used for counting task.

    :param (int) num_tiles: number of tiles to divide each image to for the puzzle
    :param (float) grayscale_probability: probability to apply grayscaling to each image
    """
    def __init__(self, num_tiles=4, grayscale_probability=0.6):
        super(CountingCollator).__init__()
        # Make sure num. tiles is a number whose square root is an integer (i.e. perfect square)
        assert num_tiles % np.sqrt(num_tiles) == 0
        self.num_tiles = num_tiles

        assert 0. <= grayscale_probability <= 1.
        self.grayscale_probability = grayscale_probability

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, _ = batch
        batch_size, img_width = x.shape[0], x.shape[-1]

        # Apply random grayscaling to prevent the model from cheating
        for i in range(batch_size):
            if np.random.rand() <= self.grayscale_probability:
                x[i] = self.rgb2grayscale(x[i]).repeat(3, 1, 1)

        # Compute the tile length and extract the tiles
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        tile_length = img_width // num_tiles_per_dimension
        tiles = []
        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
                tile_ij = x[:, :, i*tile_length: (i+1)*tile_length, j*tile_length: (j+1)*tile_length]
                tiles.append(tile_ij)
        # Tensorize the tiles
        tiles = torch.stack(tiles)

        # The tiles are tile_length x tile_length, let's downsample the original images as well
        # NOTE: Randomly pick downsampling mode in order to prevent the model from cheating
        x_downsampled = []
        for i in range(batch_size):
            mode = np.random.choice(['nearest', 'bilinear', 'bicubic', 'area'])
            # NOTE: `align_corners` only allowed with bilinear or bicubic methods
            if mode == 'bilinear' or mode == 'bicubic':
                # TODO: Check if this should be x[i] or x[:, i]
                x_i = F.interpolate(x[i].unsqueeze(0), size=(tile_length, tile_length), mode=mode, align_corners=True).squeeze(0)
            else:
                x_i = F.interpolate(x[i].unsqueeze(0), size=(tile_length, tile_length), mode=mode).squeeze(0)
            x_downsampled.append(x_i)

        x = torch.stack(x_downsampled)

        return x, tiles

    @staticmethod
    def rgb2grayscale(img):
        """
        Converts and RGB tensor image to grayscale.

        :param (torch.Tensor) img: tensor image in RGB format
        """
        r_factor, g_factor, b_factor = 0.2126, 0.7152, 0.0722
        return r_factor * img[0:1, :, :] + g_factor * img[1:2, :, :] + b_factor * img[2:3, :, :]


class ContextCollator(DataCollator):
    """
    Data collator used for context encoder task.

    :param (str) mask_method: set to 'central_blocks' to get the central square block of the image
           as a mask, set to 'random_blocks' to get the combination of multiple smaller blocks as
           a mask, or set to 'random_region' to get a randomly chosen segmentation mask from the
           PASCAL VOC 2012 dataset as a mask
    :param (tuple or float): value to fill the external regions of the images w.r.t. the mask
    """

    def __init__(self, mask_method='random_blocks', fill_value=(0.485, 0.456, 0.406)):
        super(ContextCollator).__init__()
        assert mask_method in ['central_block', 'random_blocks', 'random_region']
        self.mask_method = mask_method

        assert isinstance(fill_value, tuple) or isinstance(fill_value, int) or isinstance(fill_value, float)
        self.fill_value = fill_value

        if self.mask_method == 'random_region':
            self.segmentation_masks = self.download_segmentation_masks()

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, _ = batch
        num_channels = x.shape[1]

        if self.mask_method == 'central_block':
            mask = self.get_central_block_mask(x)
        elif self.mask_method == 'random_blocks':
            mask = self.get_random_blocks_mask(x)
        elif self.mask_method == 'random_region':
            mask = self.get_random_region_mask(x)

        x, y = (1. - mask) * x, mask * x

        # Convert `self.fill_value` to a tuple and validate it
        if isinstance(self.fill_value, int) or isinstance(self.fill_value, float):
            self.fill_value = (self.fill_value,) * num_channels
        elif len(self.fill_value) != num_channels:
            raise ValueError('The length of `fill_value` does not match the number of channels!')

        # Color empty regions with the specified fill value
        for i in range(num_channels):
            x[:, i, :, :][x[:, i, :, :] == 0.] = self.fill_value[i]

        return x, y, mask

    @staticmethod
    def get_central_block_mask(x):
        """
        Gets the central block in the image as a mask.

        :param (torch.Tensor) x: tensor image
        """
        mask = torch.zeros_like(x)
        img_height, img_width = x.shape[-2], x.shape[-1]

        block_height, block_width = img_height // 2, img_width // 2

        ycenters, xcenters = img_height // 2, img_width // 2

        ymins = ycenters - block_height // 2
        ymaxs = ycenters + block_height // 2
        xmins = xcenters - block_width // 2
        xmaxs = xcenters + block_width // 2

        mask[:, :, ymins: ymaxs, xmins: xmaxs] = 1.

        return mask

    @staticmethod
    def get_random_blocks_mask(x):
        """
        Gets randomly placed square blocks. The number of blocks per image vary from 1 to 9.

        :param (torch.Tensor) x: tensor image
        """
        mask = torch.zeros_like(x)
        batch_size, img_height, img_width = x.shape[0], x.shape[-2], x.shape[-1]

        max_num_blocks = 9
        block_height, block_width = img_height // 6, img_width // 6

        ycenters = np.random.randint(low=0, high=img_height, size=(batch_size, max_num_blocks))
        xcenters = np.random.randint(low=0, high=img_width, size=(batch_size, max_num_blocks))

        ymins = np.clip(ycenters - block_height // 2, 0, img_height)
        ymaxs = np.clip(ycenters + block_height // 2, 0, img_height)
        xmins = np.clip(xcenters - block_height // 2, 0, img_width)
        xmaxs = np.clip(xcenters + block_height // 2, 0, img_width)

        for i in range(batch_size):
            num_blocks = random.randint(1, max_num_blocks)
            for j in range(num_blocks):
                mask[:, :, ymins[i, j]: ymaxs[i, j], xmins[i, j]: xmaxs[i, j]] = 1.

        return mask

    @staticmethod
    def download_segmentation_masks():
        """Obtains random masks from the PASCAL VOC 2012 (Segmentation) Dataset."""
        segmentation_masks = []

        # Initialize the PASCAL VOC 2012 dataset for segmentation
        input_transform = transforms.Compose([transforms.Resize(1), transforms.ToTensor()])
        target_transform = transforms.Compose([transforms.ToTensor()])
        dataset = VOCSegmentation('.data/', image_set='trainval', download=True,
                                  transform=input_transform, target_transform=target_transform)
        loader = DataLoader(dataset, batch_size=1)

        for batch in tqdm(loader, desc='Loading Segmentation Masks'):
            _, mask = batch

            # Get the first (and only) example from the batch
            mask = mask[0]
            # NOTE: The masks have boundaries of 1. and inner regions of 0.5, let's all make it 1.
            mask[mask > 0.] = 1.

            # Only get masks that span up to 1/4 of the image
            if torch.mean(mask) <= 0.25:
                segmentation_masks.append(mask)

        return segmentation_masks

    def get_random_region_mask(self, x):
        """
        Chooses random masks from `self.segmentation_masks`.

        :param (torch.Tensor) x: tensor image
        """
        mask = torch.zeros_like(x)
        batch_size, img_height, img_width = x.shape[0], x.shape[-2], x.shape[-1]

        # Apply distortions in the form of affine transformations to the masks for diversification
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.RandomAffine(degrees=(-45, 45), translate=(0.25, 0.25), shear=(-10, 10),
                                    fillcolor=0),
            transforms.ToTensor(),
        ])

        for i in range(batch_size):
            segmentation_mask = random.choice(self.segmentation_masks).repeat(3, 1, 1)
            mask[i] = transform(segmentation_mask)

        return mask


class JigsawCollator(DataCollator):
    """
    Data collator used for jigsaw puzzle task.

    :param (int) num_tiles: number of tiles to divide each image to for the puzzle
    :param (int) num_permutations: number of permutations to arrange the tiles
    :param (str) permgen_method: set to 'maximal' to choose permutations based on maximal
           Hamming distance, set to 'average' for uniformly drawing permutations, or set
           to 'minimal' for minimal Hamming distance; authors indicate that 'maximal' works
           best, but is considerably slow in implementation at this point.
    :param (float) grayscale_probability: probability to apply grayscaling to each image to avoid
           shortcuts due to chromatic aberration
    :param (bool) buffer: set to True to randomly crop tiles to smaller tiles to avoid shortcuts
           due to edge continuity
    :param (bool) jitter: set to True to spatially jitter the color channels of the color images to
           avoid shortcuts due to chromatic aberration
    :param (bool) normalization: set to True to normalize the mean and standard deviation of each
           patch indepently to avoid shortcuts due to low-level statistics
    """
    def __init__(self, num_tiles=9, num_permutations=1000, permgen_method='maximal',
                 grayscale_probability=0.3, buffer=True, jitter=True, normalization=True):
        super(JigsawCollator).__init__()
        # Make sure num. tiles is a number whose square root is an integer (i.e. perfect square)
        assert num_tiles % np.sqrt(num_tiles) == 0
        self.num_tiles = num_tiles
        self.num_permutations = num_permutations

        assert 0. <= grayscale_probability <= 1.
        self.grayscale_probability = grayscale_probability

        self.buffer = buffer
        self.jitter = jitter
        self.normalization = normalization

        # Randomly pick `k` permutations for tile configurations (i.e. the permutation set)
        # NOTE: We assign indices to each tile configuration with the list `self.permutations
        self.permutations = self.generate_permutation_set(num_tiles=num_tiles,
                                                          num_permutations=num_permutations,
                                                          method=permgen_method)

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, _ = batch
        batch_size, img_height, img_width = x.shape[0], x.shape[-2], x.shape[-1]

        # Make sure that the image is square; jigsaw puzzles are for squares
        assert img_height == img_width

        # Apply random grayscaling to avoid shortcuts due to chromatic aberration
        for i in range(batch_size):
            if np.random.rand() <= self.grayscale_probability:
                x[i] = self.rgb2grayscale(x[i]).repeat(3, 1, 1)

        # Compute the tile length and extract the tiles
        tiles = []
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        tile_length = img_width // num_tiles_per_dimension

        buffer = int(tile_length * 0.1)

        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
                if self.buffer:
                    tile_ij = torch.empty(batch_size, x.shape[1], tile_length - buffer, tile_length - buffer)
                else:
                    tile_ij = x[:, :, i * tile_length: (i + 1) * tile_length, j * tile_length: (j + 1) * tile_length]

                for k in range(batch_size):
                    num_channels = tile_ij.shape[1]

                    # Leave a random gap between tiles to avoid shortcuts due to edge continuity
                    if self.buffer:
                        # Randomly sample values that sum up to `buffer` for cropping
                        # NOTE: The summing up helps keep the same cropped tile size
                        buffer_x1, buffer_x2 = np.random.multinomial(buffer, [0.5, 0.5])
                        buffer_y1, buffer_y2 = np.random.multinomial(buffer, [0.5, 0.5])

                        tile_x1 = i * tile_length + buffer_x1
                        tile_x2 = (i + 1) * tile_length - buffer_x2
                        tile_y1 = j * tile_length + buffer_y1
                        tile_y2 = (j + 1) * tile_length - buffer_y2

                        tile_ij[k] = x[k, :, tile_x1: tile_x2, tile_y1: tile_y2]

                    # Apply random spatial jitter to avoid shortcuts due to chromatic aberration
                    if self.jitter and num_channels == 3:
                        tile_ij[k] = torch.stack([torch.roll(tile_ij[k, 0], random.randint(-2, 2)),
                                                  torch.roll(tile_ij[k, 1], random.randint(-2, 2)),
                                                  torch.roll(tile_ij[k, 2], random.randint(-2, 2))])

                    # Apply normalization to avoid shortcuts due to low-level statistics
                    if self.normalization:
                        # NOTE: The normalization is independently applied to each image patch
                        normalization = transforms.Normalize([tile_ij[k, c].mean() for c in range(num_channels)],
                                                             [tile_ij[k, c].std() for c in range(num_channels)])
                        tile_ij[k] = normalization(tile_ij[k])

                tiles.append(tile_ij)

        # Tensorize the tiles
        tiles = torch.stack(tiles)

        # Shuffle tiles according to a randomly drawn in sequence in the permutations list
        y = []
        for i in range(batch_size):
            permutation_index = np.random.randint(0, self.num_permutations)
            permutation = torch.tensor(self.permutations[permutation_index])

            tiles[:, i, :, :] = tiles[permutation, i, :, :]
            y.append(permutation_index)

        y = torch.tensor(y).long()

        return tiles, y

    @staticmethod
    def rgb2grayscale(img):
        """
        Converts and RGB tensor image to grayscale.

        :param (torch.Tensor) img: tensor image in RGB format
        """
        r_factor, g_factor, b_factor = 0.2126, 0.7152, 0.0722
        return r_factor * img[0:1, :, :] + g_factor * img[1:2, :, :] + b_factor * img[2:3, :, :]

    @staticmethod
    def generate_permutation_set(num_tiles, num_permutations, method='maximal'):
        """
        Function for generating a permutation set based on a Hamming distance based objective.
        Follows the pseudocode given by the authors in the paper.
        """
        if method not in ['maximal', 'average', 'minimal']:
            raise ValueError('The specified method=%s is not recognized!' % method)

        # Initialize the output set
        permutations = []

        # Get all permutations
        tile_positions = list(range(num_tiles))
        all_permutations = list(itertools.permutations(tile_positions))

        # Convert `all_permutations` to a 2D matrix
        all_permutations = np.array(all_permutations).T  # num_tiles x (num_tiles)! matrix

        # Uniformly sample out of (num_tiles)! indices to initialize
        current_index = random.randint(0, np.math.factorial(num_tiles) - 1)

        for i in tqdm(range(1, num_permutations + 1), desc='Generating Permutation Set'):
            # Add permutation at current index to the output set
            permutations.append(tuple(all_permutations[:, current_index]))
            # Remove permutation at current index from `all_permutations`
            all_permutations = np.delete(all_permutations, current_index, axis=1)

            # Uniformly sample all the way if `method` is average and skip computations
            if method == 'average':
                current_index = random.randint(0, np.math.factorial(num_tiles) - i)
                continue

            # Compute the Hamming distance matrix
            distances = np.empty((i, np.math.factorial(num_tiles) - i))

            # TODO: This nested loop takes a very long time!
            for j in range(i):
                for k in range(np.math.factorial(num_tiles) - i):
                    distances[j, k] = hamming(permutations[j], all_permutations[:, k])

            # Convert the matrix into a (summed) row vector of shape 1 x ((num_tiles)! - i)
            distances = np.matmul(np.ones((1, i)), distances)

            # Choose the next permutation s.t. it maximizes the objective
            if method == 'maximal':
                current_index = np.argmax(distances)
            elif method == 'minimal':
                current_index = np.argmin(distances)

        # Compute the minimum hamming distance in the generated permutation set
        distances_ = []
        for i in range(num_permutations):
            for j in range(num_permutations):
                if i != j:
                    distances_.append(hamming(np.array(permutations[i]), np.array(permutations[j])))

        min_distance = min(distances_)
        print('Minimum hamming distance is chosen as %0.4f' % min_distance)

        return permutations
