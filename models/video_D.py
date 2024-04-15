# noinspection PyShadowingNames
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import constants as c
from registry import register_model


@register_model
def discriminator():
    return DiscriminatorModel(c.TRAIN_HEIGHT,
                              c.TRAIN_WIDTH,
                              c.SCALE_CONV_FMS_D,
                              c.SCALE_KERNEL_SIZES_D,
                              c.SCALE_FC_LAYER_SIZES_D)


@register_model
def generator():
    return GeneratorModel(
                          c.TRAIN_HEIGHT,
                          c.TRAIN_WIDTH,
                          c.FULL_HEIGHT,
                          c.FULL_WIDTH,
                          c.SCALE_FMS_G,
                          c.SCALE_KERNEL_SIZES_G)


class DiscriminatorModel:
    def __init__(self, height, width, scale_conv_layer_fms, scale_kernel_sizes, scale_fc_layer_sizes):
        """
        Initializes a DiscriminatorModel.

        @param height: The height of the input images.
        @param width: The width of the input images.
        @param scale_conv_layer_fms: The number of feature maps in each convolutional layer of each
                                     scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.
        @param scale_fc_layer_sizes: The number of nodes in each fully-connected layer of each scale
                                     network.

        @type height: int
        @type width: int
        @type scale_conv_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        @type scale_fc_layer_sizes: list<list<int>>
        """
        self.height = height
        self.width = width
        self.scale_conv_layer_fms = scale_conv_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_fc_layer_sizes = scale_fc_layer_sizes
        self.num_scale_nets = len(scale_conv_layer_fms)
        self.scale_nets = []
        self.scale_preds = []

        for scale_num in range(self.num_scale_nets):
            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
            scale_net = DScaleModel(scale_num,
                                    int(self.height * scale_factor),
                                    int(self.width * scale_factor),
                                    self.scale_conv_layer_fms[scale_num],
                                    self.scale_kernel_sizes[scale_num],
                                    self.scale_fc_layer_sizes[scale_num])
            self.scale_nets.append(scale_net)

        self.optimizer = optim.SGD([param for scale_net in self.scale_nets for param in scale_net.parameters()],
                                   lr=c.LRATE_D)
        self.global_step = 0

    def build_feed_dict(self, input_frames, gt_output_frames, generator):
        """
        Builds a feed_dict with resized inputs and outputs for each scale network.

        @param input_frames: An array of shape [batch_size x self.height x self.width x (3 * HIST_LEN)], The frames to
                             use for generation.
        @param gt_output_frames: An array of shape [batch_size x self.height x self.width x 3], The
                                 ground truth outputs for each sequence in input_frames.
        @param generator: The generator model.

        @return: The feed_dict needed to run this network, all scale_nets, and the generator
                 predictions.
        """
        feed_dict = {}
        batch_size = gt_output_frames.shape[0]

        g_feed_dict = {generator.input_frames_train: input_frames, generator.gt_frames_train: gt_output_frames}
        g_scale_preds = generator(g_feed_dict)  # Assuming generator is a PyTorch model and returns predictions

        for scale_num, scale_net in enumerate(self.scale_nets):
            scaled_height = int(self.height * (1. / 2 ** ((self.num_scale_nets - 1) - scale_num)))
            scaled_width = int(self.width * (1. / 2 ** ((self.num_scale_nets - 1) - scale_num)))
            scaled_gt_output_frames = torch.nn.functional.interpolate(gt_output_frames,
                                                                      size=(scaled_height, scaled_width),
                                                                      mode='bilinear', align_corners=False)
            scaled_input_frames = torch.cat([g_scale_preds[scale_num], scaled_gt_output_frames])

            feed_dict[scale_net.input_frames] = scaled_input_frames

        targets = torch.cat([torch.zeros(batch_size, 1), torch.ones(batch_size, 1)])
        feed_dict['targets'] = targets

        return feed_dict

    def train_step(self, batch, generator):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape [BATCH_SIZE x self.height x self.width x (3 * (HIST_LEN + 1))]. The input and
                      output frames, concatenated along the channel axis.
        @param generator: The generator model.

        @return: The global step.
        """
        input_frames = batch[:, :, :, :-3]
        gt_output_frames = batch[:, :, :, -3:]

        feed_dict = self.build_feed_dict(input_frames, gt_output_frames, generator)

        # Assuming DScaleModel has a method named "forward" to make predictions
        scale_preds = [scale_net(scale_net.input_frames) for scale_net in self.scale_nets]

        # Assuming adv_loss is a function that calculates adversarial loss
        global_loss = adv_loss(scale_preds, feed_dict['targets'])

        self.optimizer.zero_grad()
        global_loss.backward()
        self.optimizer.step()

        self.global_step += 1

        if self.global_step % c.STATS_FREQ == 0:
            print('DiscriminatorModel: step %d | global loss: %f' % (self.global_step, global_loss.item()))

        return self.global_step


class DScaleModel(nn.Module):
    """
    DScaleModel is a PyTorch neural network that discriminates whether an input video frame is a real-world image or one generated by a generator network.
    Multiple instances of this model can be used together to make predictions on frames at increasing scales.
    """

    def __init__(self, scale_index, height, width, conv_layer_fms, kernel_sizes, fc_layer_sizes):
        """
        Initializes the DScaleModel.

        :param scale_index: The index number of this height in the DiscriminatorModel.
        :param height: The height of the input images.
        :param width: The width of the input images.
        :param conv_layer_fms: The number of output feature maps for each convolution.
        :param kernel_sizes: The size of the kernel for each convolutional layer.
        :param fc_layer_sizes: The number of nodes in each fully-connected layer.
        """
        super(DScaleModel, self).__init__()
        assert len(kernel_sizes) == len(conv_layer_fms) - 1, 'len(kernel_sizes) must equal len(conv_layer_fms) - 1'

        self.scale_index = scale_index
        self.height = height
        self.width = width
        self.conv_layer_fms = conv_layer_fms
        self.kernel_sizes = kernel_sizes
        self.fc_layer_sizes = fc_layer_sizes

        self.define_graph()

    def define_graph(self):
        """
        Sets up the model graph in PyTorch.
        """

        # Convolutional layers
        layers = []
        last_out_channels = self.conv_layer_fms[0]
        for i, (kernel_size, out_fm) in enumerate(zip(self.kernel_sizes, self.conv_layer_fms[:-1])):
            layers.append(nn.Conv2d(last_out_channels, out_fm, kernel_size=(kernel_size, kernel_size), padding=1))
            layers.append(nn.ReLU())
            last_out_channels = out_fm
            last_out_height = (self.height - kernel_size + 2 * 1) // 2 + 1
            last_out_width = (self.width - kernel_size + 2 * 1) // 2 + 1
            self.height = last_out_height
            self.width = last_out_width

        # Pooling layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Fully connected layers
        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.ModuleList()
        last_size = (self.height * self.width * last_out_channels)

        for i, fc_size in enumerate(self.fc_layer_sizes[:-1]):
            self.fc_layers.append(nn.Linear(last_size, fc_size))
            last_size = fc_size

        self.fc_layers.append(nn.Linear(last_size, self.fc_layer_sizes[-1]))
        self.fc_layers.append(nn.Sigmoid())

    def forward(self, x):
        """
        Runs the input through the network to generate a prediction from 0 (generated img) to 1 (real img).

        :param x: The input tensor of shape [batch_size, channels, height, width].
        :return: A tensor of predictions of shape [batch_size, 1].
        """
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for dense layers
        for fc in self.fc_layers:
            x = F.relu(fc(x)) if fc != self.fc_layers[-1] else x = self.fc_layers[-1](x)
            x = torch.clamp(x, min=0.1, max=0.9)  # Clip for stability
        return x


def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generatoDScaleModelr passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    loss = -(targets * torch.log10(preds) + (1 - targets) * torch.log10(1 - preds))
    return torch.squeeze(loss).sum()


def adv_loss(preds, labels):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.

    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).

    @return: The adversarial loss.
    """
    bce_loss = torch.nn.BCELoss()

    scale_losses = [bce_loss(pred, labels) for pred in preds]
    total_loss = torch.mean(torch.stack(scale_losses))

    return total_loss
