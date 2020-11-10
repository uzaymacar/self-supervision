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
    Data collator used for cutout.
    """
    def __init__(self, num_rotation_classes=4):
        super(RotationCollator).__init__()
        self.rotation_degrees = np.linspace(0, 360, num_rotation_classes+1).tolist()[:-1]
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
            y[i] = torch.tensor(self.rotation_degrees.index(theta)).to(y.dtype)

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