import torch
import torch.nn.functional as F
import numpy as np
import random
import itertools


class DataCollator(object):
    def __init__(self):
        pass

    def __call__(self, examples):
        pass

    def _preprocess_batch(self, examples):
        """
        Preprocesses tensors: returns the current examples as batch.
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
            permutation = torch.randperm(batch_size)
            x, y = x[permutation], y[permutation]

        return x, y

    @staticmethod
    def get_rotation_matrix(theta, mode='degrees'):
        assert mode in ['degrees', 'radians']

        if mode == 'degrees':
            theta *= np.pi / 180

        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def rotate_image(self, x, theta):
        """https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports"""
        dtype = x.dtype
        rotation_matrix = self.get_rotation_matrix(theta=theta, mode='degrees')[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rotation_matrix, x.shape, align_corners=True).type(dtype)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


class CountingCollator(DataCollator):
    """
    Data collator used for counting task.
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
        r_factor, g_factor, b_factor = 0.2126, 0.7152, 0.0722
        return r_factor * img[0:1, :, :] + g_factor * img[1:2, :, :] + b_factor * img[2:3, :, :]


class ContextCollator(DataCollator):
    """
    Data collator used for context encoder task.
    """
    def __init__(self, mask_method='central_block', fill_value=(0.485, 0.456, 0.406)):
        super(ContextCollator).__init__()
        # Make sure num. tiles is a number whose square root is an integer (i.e. perfect square)
        assert mask_method in ['central_block', 'random_blocks', 'random_region']
        self.mask_method = mask_method

        if isinstance(fill_value, int):
            fill_value = (fill_value, ) * 3
        self.fill_value = fill_value

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, _ = batch

        if self.mask_method == 'central_block':
            mask = self.get_central_mask(x)
        else:
            raise NotImplementedError('The specified mask_method=%s is not yet implemented!' % self.mask_method)

        x, y = (1. - mask) * x, mask * x
        # TODO: Incorporate fill_value here
        return x, y, mask

    @staticmethod
    def get_central_mask(x):
        mask = torch.zeros_like(x)

        img_height, img_width = x.shape[-2], x.shape[-1]
        region_height, region_width = img_height // 2, img_width // 2
        ver0, hor0 = (img_height - region_height) // 2, (img_width - region_width) // 2

        mask[:, :, ver0: ver0 + region_height, hor0: hor0 + region_width] = 1.
        return mask


class JigsawCollator(DataCollator):
    """
    Data collator used for jigsaw puzzle task.
    """
    def __init__(self, num_tiles=9, num_permutations=1000, grayscale_probability=0.3):
        super(JigsawCollator).__init__()
        # Make sure num. tiles is a number whose square root is an integer (i.e. perfect square)
        assert num_tiles % np.sqrt(num_tiles) == 0
        self.num_tiles = num_tiles
        self.num_permutations = num_permutations

        assert 0. <= grayscale_probability <= 1.
        self.grayscale_probability = grayscale_probability

        tile_positions = list(range(0, num_tiles))
        self.permutations_set = random.sample(list(itertools.permutations(tile_positions)), k=num_permutations)

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

        # Shuffle tiles according to a randomly drawn in sequence in the permutations set
        y = []
        for i in range(batch_size):
            permutation_index = np.random.randint(0, self.num_permutations)
            permutation = torch.tensor(self.permutations_set[permutation_index])

            # TODO: Check if this should be tiles[:, i, :, :] or tiles[i, :, :, :]
            tiles[:, i, :, :] = tiles[permutation, i, :, :]
            y.append(permutation_index)

        y = torch.tensor(y).long()

        return tiles, y

    @staticmethod
    def rgb2grayscale(img):
        r_factor, g_factor, b_factor = 0.2126, 0.7152, 0.0722
        return r_factor * img[0:1, :, :] + g_factor * img[1:2, :, :] + b_factor * img[2:3, :, :]


class ColorizationCollator(DataCollator):
    """
    Data collator used for colorization task.
    """
    def __init__(self):
        super(ColorizationCollator).__init__()

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, _ = batch

        # Extract only the lightness (L) color channel, original will be used as pseudo-labels
        x_l = x[:, torch.tensor([0, 0, 0]), :, :]
        assert x_l.shape == x.shape

        return x_l, x





