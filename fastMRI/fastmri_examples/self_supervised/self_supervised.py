import torch
import torch.nn as nn
from fastmri.data import transforms as fastmri_transforms


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Each RB comprised of two convolutional layers.  All layers had a kernel size of 3×3 and 64 channels.

    The first layer is followed by a ReLU activation.  The second is followed by a constant multiplication, with factor
    (by default) equal to 0.1.
    """
    def __init__(self, in_channels=64, 
                out_channels=64, stride=1, 
                const_multiple=0.1, 
                downsample=None,
                lr=0.001,
                lr_step_size=40,
                lr_gamma=0.1,
                weight_decay=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self._scalar_mult_2 = const_multiple

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out *= self._scalar_mult_2
        # decide not to add itself in RB
        identity = identity if self.downsample is None else self.downsample(x)
        out += identity 
        out = self.relu(out)
        return out


class MriSelfSupervised(nn.Module):

    def __init__(self, input_channels=4, output_channels=64):
        super(MriSelfSupervised, self).__init__()
        # layer of input and output convolution layers
        # 15 residual blocks (RB) with skip connections
            # Each RB comprised of two convolutional layers
                # first layer is followed by a rectified linear unit (ReLU)
                # second layer is followed by a constant multiplication layer, with factor C = 0.1 (55).
                # All layers had a kernel size of 3×3 and 64 channels
        self.mri_relu = nn.ReLU(inplace=True)
        self.mri_conv1 = nn.Conv2d(input_channels, output_channels, stride=1, padding=1, kernel_size=3)
        self.mri_bn = nn.BatchNorm2d(output_channels)
        
        self.residual_blocks = nn.ModuleList()
        for i in range(15):
          self.residual_blocks.append(ResidualBlock())

        #self.mri_out1 = nn.Conv2d(output_channels,input_channels, kernel_size=3, stride=1, padding=1)
        #self.mri_out2 = nn.Conv2d(input_channels,input_channels, kernel_size=3, stride=1, padding=1)
        self.mri_out1 = nn.Conv2d(output_channels,4, kernel_size=3, stride=1, padding=1)
        self.mri_out2 = nn.Conv2d(4,2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            A volume for an MRI image.

        Returns
        -------

        """
        #print(x.shape)
        out = self.mri_relu(x)
        out = self.mri_conv1(out)
        #out = self.mri_bn(out)
        out = self.mri_relu(out)
        for rb in self.residual_blocks:
            out = rb(out)

        out = self.mri_out1(out)
        
        out = out + x # add itself: this could be important
        out = self.mri_out2(out)
        #out, _, _ = fastmri_transforms.normalize_instance(out, eps=1e-11)
        #out = out.clamp(-6, 6)
        
        return out


