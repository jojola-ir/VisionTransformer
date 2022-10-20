import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits image into patches and embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (m*m).

    patch_size : int
        Size of a patch (n*n).

    input_chan : int
        Number of input channels.

    embed_dim : int
        Dimension of the embedding.

    Attributes
    ----------
    nb_patches : int
        Number of patches of the image.

    embed : nn.Conv2d
        Convolutional layer that splits image to patches and embed them.
    """
    def __init__(self, img_size, patch_size, input_chan=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (input_chan // patch_size) ** 2

        self.embed = nn.Conv2d(
            input_chan,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """Runs forward pass.

        Parameters
        ----------
        x : torch.Tensor
           Shape : (n_samples, input_chan, img_size, img_size)

        Returns
        -------
        torch.Tensor
            Shape : (n_samples, n_patches, embed_dim)
        """
        x = self.embed(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        print(x.shape)
        x = x.flatten(2) # (n_sample, embed_dim, n_patches)
        print(x.shape)
        x = x.transpose(1, 2) # (n_sample, n_patches, embed_dim)
        print(x.shape)

        return x


if __name__ == '__main__':

    vec = torch.ones(1, 3, 8, 8)
    PE = PatchEmbed(vec.shape[2], 2)(vec)