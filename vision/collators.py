import torch
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, num_rotations=4):
        super(RotationCollator).__init__()
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
        # We are excluding the last item, 360 degrees, since it is equivalent to 0 degrees

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        x, y = batch

        batch_size = x.shape[0]
        for i in range(batch_size):
            theta = np.random.choice(self.rotation_degrees, size=1)[0]
            x[i] = self.rotate_image(x[i].unsqueeze(0), theta=theta).squeeze(0)
            # TODO: Turns out we can tensorize/vectorize the above!
            y[i] = torch.tensor(self.rotation_degrees.index(theta)).long()  # to(y.dtype)

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
    def __init__(self, num_tiles=4):
        super(CountingCollator).__init__()
        # Make sure num. tiles is a number whose square root is an integer (i.e. perfect square)
        assert num_tiles % np.sqrt(num_tiles) == 0
        self.num_tiles = num_tiles

    def __call__(self, examples):
        """
        Makes the class callable, just like a function.
        """
        batch = self._preprocess_batch(examples)
        originals, _ = batch
        # Create the other full-size random images to be contrasted to the originals
        others = torch.cat([originals[1:], originals[0:1]], dim=0)

        # Compute the tile length and extract the tiles
        image_width = originals[0].shape[-1]
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        tile_length = image_width // num_tiles_per_dimension
        tiles = []
        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
                tile_ij = originals[:, :, i*tile_length: (i+1)*tile_length, j*tile_length: (j+1)*tile_length]
                tiles.append(tile_ij)
        # Tensorize the tiles
        tiles = torch.stack(tiles)

        # The tiles are tile_length x tile_length, let's resize the other image batches as well
        # NOTE: Randomly pick downsampling mode in order to prevent the model from cheating
        mode = np.random.choice(['nearest', 'bilinear', 'bicubic', 'area'], size=1)
        if mode == 'bilinear' or mode == 'bicubic':
            # NOTE: `align_corners` only allowed with bilinear or bicubic methods
            originals = F.interpolate(originals, size=(tile_length, tile_length), mode=mode, align_corners=True)
            others = F.interpolate(others, size=(tile_length, tile_length), mode=mode, align_corners=True)
        else:
            originals = F.interpolate(originals, size=(tile_length, tile_length), mode=mode)
            others = F.interpolate(others, size=(tile_length, tile_length), mode=mode)

        return originals, tiles, others
