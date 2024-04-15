from scipy.misc import imsave
import os

from torch.optim import optimizer

import constants as c
import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from copy import deepcopy

class GeneratorModel(nn.Module):
    def __init__(self, height_train, width_train, height_test, width_test, scale_layer_fms, scale_kernel_sizes, adversarial):
        super(GeneratorModel, self).__init__()
        self.height_train = height_train
        self.width_train = width_train
        self.height_test = height_test
        self.width_test = width_test
        self.scale_layer_fms = scale_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.num_scale_nets = len(scale_layer_fms)
        self.adversarial = adversarial
        # other initialization here

    def define_graph(self):
        # Define the model layers and setup the optimizer here
        # This will include setting up the model architecture using nn.Module, defining the loss function, and setting up the optimizer

    def train_step(self, batch, discriminator=None):
        input_frames = torch.from_numpy(batch[:, :, :, :-3]).float()
        gt_frames = torch.from_numpy(batch[:, :, :, -3:]).float()

        optimizer.zero_grad()  # Clear the gradients

        if self.adversarial:
            # Run the generator first to get generated frames
            scale_preds = self.forward(input_frames)

            # Run the discriminator nets on those frames to get predictions
            d_scale_preds = []
            for scale_num, gen_frames in enumerate(scale_preds):
                # Run the discriminator model to get predictions d_scale_preds
                # Add the predictions to the d_scale_preds list

        logits, global_loss, psnr_error, sharpdiff_error = self.calculate_loss_and_error(scale_preds[-1], gt_frames)
        global_loss.backward()  # Backpropagation
        optimizer.step()  # Update the model weights

        # Other user outputs and saving logic

    def test_batch(self, batch, global_step, num_rec_out=1, save_imgs=True):
        input_frames = torch.from_numpy(batch[:, :, :, :3 * c.HIST_LEN]).float()
        gt_frames = torch.from_numpy(batch[:, :, :, 3 * c.HIST_LEN:]).float()

        rec_preds = []
        # Generate num_rec_out recursive predictions
        working_input_frames = input_frames.clone()
        for rec_num in range(num_rec_out):
            preds = self.forward(working_input_frames)
            rec_preds.append(preds)
            working_input_frames = torch.cat((working_input_frames[:, :, :, 3:], preds), dim=3)

        # Save images and other output logic


# Example usage:
# Initialize the model with the appropriate parameters
# generator = GeneratorModel(height_train=256, width_train=256, height_test=256, width_test=256,
#                            scale_layer_fms=[[...], [...]], scale_kernel_sizes=[[...], [...]])


# Define your loss function, optimizer, and any other necessary components
# Train and test the model with your data
# noinspection PyShadowingNames
class GeneratorModel:
    def __init__(self, session, summary_writer, height_train, width_train, height_test,
                 width_test, scale_layer_fms, scale_kernel_sizes):
        self.sess = session
        self.summary_writer = summary_writer
        self.height_train = height_train
        self.width_train = width_train
        self.height_test = height_test
        self.width_test = width_test
        self.scale_layer_fms = scale_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.num_scale_nets = len(scale_layer_fms)

        self.define_graph()

    # noinspection PyAttributeOutsideInit
    def define_graph(self):
        """
        Sets up the model graph in TensorFlow.
        """
        with tf.name_scope('generator'):
            ##
            # Data
            ##

            with tf.name_scope('data'):
                self.input_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, 3 * c.HIST_LEN])
                self.gt_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, 3])

                self.input_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, 3 * c.HIST_LEN])
                self.gt_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, 3])

                # use variable batch_size for more flexibility
                self.batch_size_train = tf.shape(self.input_frames_train)[0]
                self.batch_size_test = tf.shape(self.input_frames_test)[0]

            ##
            # Scale network setup and calculation
            ##

            self.summaries_train = []
            self.scale_preds_train = []  # the generated images at each scale
            self.scale_gts_train = []  # the ground truth images at each scale
            self.d_scale_preds = []  # the predictions from the discriminator model

            self.summaries_test = []
            self.scale_preds_test = []  # the generated images at each scale
            self.scale_gts_test = []  # the ground truth images at each scale

            for scale_num in range(self.num_scale_nets):
                with tf.name_scope('scale_' + str(scale_num)):
                    with tf.name_scope('setup'):
                        ws = []
                        bs = []

                        # create weights for kernels
                        for i in range(len(self.scale_kernel_sizes[scale_num])):
                            ws.append(w([self.scale_kernel_sizes[scale_num][i],
                                         self.scale_kernel_sizes[scale_num][i],
                                         self.scale_layer_fms[scale_num][i],
                                         self.scale_layer_fms[scale_num][i + 1]]))
                            bs.append(b([self.scale_layer_fms[scale_num][i + 1]]))

                    with tf.name_scope('calculation'):
                        def calculate(height, width, inputs, gts, last_gen_frames):
                            # scale inputs and gts
                            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                            scale_height = int(height * scale_factor)
                            scale_width = int(width * scale_factor)

                            inputs = tf.image.resize_images(inputs, [scale_height, scale_width])
                            scale_gts = tf.image.resize_images(gts, [scale_height, scale_width])

                            # for all scales but the first, add the frame generated by the last
                            # scale to the input
                            if scale_num > 0:
                                last_gen_frames = tf.image.resize_images(
                                    last_gen_frames,[scale_height, scale_width])
                                inputs = tf.concat(3, [inputs, last_gen_frames])

                            # generated frame predictions
                            preds = inputs

                            # perform convolutions
                            with tf.name_scope('convolutions'):
                                for i in range(len(self.scale_kernel_sizes[scale_num])):
                                    # Convolve layer
                                    preds = tf.nn.conv2d(
                                        preds, ws[i], [1, 1, 1, 1], padding=c.PADDING_G)

                                    # Activate with ReLU (or Tanh for last layer)
                                    if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                                        preds = tf.nn.tanh(preds + bs[i])
                                    else:
                                        preds = tf.nn.relu(preds + bs[i])

                            return preds, scale_gts

                        ##
                        # Perform train calculation
                        ##

                        # for all scales but the first, add the frame generated by the last
                        # scale to the input
                        if scale_num > 0:
                            last_scale_pred_train = self.scale_preds_train[scale_num - 1]
                        else:
                            last_scale_pred_train = None

                        # calculate
                        train_preds, train_gts = calculate(self.height_train,
                                                           self.width_train,
                                                           self.input_frames_train,
                                                           self.gt_frames_train,
                                                           last_scale_pred_train)
                        self.scale_preds_train.append(train_preds)
                        self.scale_gts_train.append(train_gts)

                        # We need to run the network first to get generated frames, run the
                        # discriminator on those frames to get d_scale_preds, then run this
                        # again for the loss optimization.
                        if c.ADVERSARIAL:
                            self.d_scale_preds.append(tf.placeholder(tf.float32, [None, 1]))

                        ##
                        # Perform test calculation
                        ##

                        # for all scales but the first, add the frame generated by the last
                        # scale to the input
                        if scale_num > 0:
                            last_scale_pred_test = self.scale_preds_test[scale_num - 1]
                        else:
                            last_scale_pred_test = None

                        # calculate
                        test_preds, test_gts = calculate(self.height_test,
                                                         self.width_test,
                                                         self.input_frames_test,
                                                         self.gt_frames_test,
                                                         last_scale_pred_test)
                        self.scale_preds_test.append(test_preds)
                        self.scale_gts_test.append(test_gts)

            with tf.name_scope('train'):
                # global loss is the combined loss from every scale network
                self.global_loss = combined_loss(self.scale_preds_train,
                                                 self.scale_gts_train,
                                                 self.d_scale_preds)
                self.global_step = tf.Variable(0, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=c.LRATE_G, name='optimizer')
                self.train_op = self.optimizer.minimize(self.global_loss,
                                                        global_step=self.global_step,
                                                        name='train_op')

                # train loss summary
                loss_summary = tf.scalar_summary('train_loss_G', self.global_loss)
                self.summaries_train.append(loss_summary)

            with tf.name_scope('error'):
                # error computation
                # get error at largest scale
                self.psnr_error_train = psnr_error(self.scale_preds_train[-1],
                                                   self.gt_frames_train)
                self.sharpdiff_error_train = sharp_diff_error(self.scale_preds_train[-1],
                                                              self.gt_frames_train)
                self.psnr_error_test = psnr_error(self.scale_preds_test[-1],
                                                  self.gt_frames_test)
                self.sharpdiff_error_test = sharp_diff_error(self.scale_preds_test[-1],
                                                             self.gt_frames_test)
                # train error summaries
                summary_psnr_train = tf.scalar_summary('train_PSNR',
                                                       self.psnr_error_train)
                summary_sharpdiff_train = tf.scalar_summary('train_SharpDiff',
                                                            self.sharpdiff_error_train)
                self.summaries_train += [summary_psnr_train, summary_sharpdiff_train]

                # test error
                summary_psnr_test = tf.scalar_summary('test_PSNR',
                                                      self.psnr_error_test)
                summary_sharpdiff_test = tf.scalar_summary('test_SharpDiff',
                                                           self.sharpdiff_error_test)
                self.summaries_test += [summary_psnr_test, summary_sharpdiff_test]

            # add summaries to visualize in TensorBoard
            self.summaries_train = tf.merge_summary(self.summaries_train)
            self.summaries_test = tf.merge_summary(self.summaries_test)

    def train_step(self, batch, discriminator=None):
        input_frames = batch[:, :, :, :-3]
        gt_frames = batch[:, :, :, -3:]

        feed_dict = {self.input_frames_train: input_frames, self.gt_frames_train: gt_frames}

        if c.ADVERSARIAL:
            # Run the generator first to get generated frames
            scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

            # Run the discriminator nets on those frames to get predictions
            d_feed_dict = {}
            for scale_num, gen_frames in enumerate(scale_preds):
                d_feed_dict[discriminator.scale_nets[scale_num].input_frames] = gen_frames
            d_scale_preds = self.sess.run(discriminator.scale_preds, feed_dict=d_feed_dict)

            # Add discriminator predictions to the
            for i, preds in enumerate(d_scale_preds):
                feed_dict[self.d_scale_preds[i]] = preds

        _, global_loss, global_psnr_error, global_sharpdiff_error, global_step, summaries = \
            self.sess.run([self.train_op,
                           self.global_loss,
                           self.psnr_error_train,
                           self.sharpdiff_error_train,
                           self.global_step,
                           self.summaries_train],
                          feed_dict=feed_dict)

        ##
        # User output
        ##
        if global_step % c.STATS_FREQ == 0:
            print 'GeneratorModel : Step ', global_step
            print '                 Global Loss    : ', global_loss
            print '                 PSNR Error     : ', global_psnr_error
            print '                 Sharpdiff Error: ', global_sharpdiff_error
        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print 'GeneratorModel: saved summaries'
        if global_step % c.IMG_SAVE_FREQ == 0:
            print '-' * 30
            print 'Saving images...'

            # if not adversarial, we didn't get the preds for each scale net before for the
            # discriminator prediction, so do it now
            if not c.ADVERSARIAL:
                scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

            # re-generate scale gt_frames to avoid having to run through TensorFlow.
            scale_gts = []
            for scale_num in range(self.num_scale_nets):
                scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                scale_height = int(self.height_train * scale_factor)
                scale_width = int(self.width_train * scale_factor)

                # resize gt_output_frames for scale and append to scale_gts_train
                scaled_gt_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, 3])
                for i, img in enumerate(gt_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize
                    # to [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, 3])
                    scaled_gt_frames[i] = (resized_frame - 0.5) * 2
                scale_gts.append(scaled_gt_frames)

            # for every clip in the batch, save the inputs, scale preds and scale gts
            for pred_num in range(len(input_frames)):
                pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Step_' + str(global_step),
                                                  str(pred_num)))

                # save input images
                for frame_num in range(c.HIST_LEN):
                    img = input_frames[pred_num, :, :, (frame_num * 3):((frame_num + 1) * 3)]
                    imsave(os.path.join(pred_dir, 'input_' + str(frame_num) + '.png'), img)

                # save preds and gts at each scale
                # noinspection PyUnboundLocalVariable
                for scale_num, scale_pred in enumerate(scale_preds):
                    gen_img = scale_pred[pred_num]

                    path = os.path.join(pred_dir, 'scale' + str(scale_num))
                    gt_img = scale_gts[scale_num][pred_num]

                    imsave(path + '_gen.png', gen_img)
                    imsave(path + '_gt.png', gt_img)

            print 'Saved images!'
            print '-' * 30

        return global_step

    def test_batch(self, batch, global_step, num_rec_out=1, save_imgs=True):
        if num_rec_out < 1:
            raise ValueError('num_rec_out must be >= 1')

        print '-' * 30
        print 'Testing:'

        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :3 * c.HIST_LEN]
        gt_frames = batch[:, :, :, 3 * c.HIST_LEN:]

        ##
        # Generate num_rec_out recursive predictions
        ##

        working_input_frames = deepcopy(input_frames)  # input frames that will shift w/ recursion
        rec_preds = []
        rec_summaries = []
        for rec_num in range(num_rec_out):
            working_gt_frames = gt_frames[:, :, :, 3 * rec_num:3 * (rec_num + 1)]

            feed_dict = {self.input_frames_test: working_input_frames,
                         self.gt_frames_test: working_gt_frames}
            preds, psnr, sharpdiff, summaries = self.sess.run([self.scale_preds_test[-1],
                                                               self.psnr_error_test,
                                                               self.sharpdiff_error_test,
                                                               self.summaries_test],
                                                              feed_dict=feed_dict)

            # remove first input and add new pred as last input
            working_input_frames = np.concatenate(
                [working_input_frames[:, :, :, 3:], preds], axis=3)

            # add predictions and summaries
            rec_preds.append(preds)
            rec_summaries.append(summaries)

            print 'Recursion ', rec_num
            print 'PSNR Error     : ', psnr
            print 'Sharpdiff Error: ', sharpdiff

        # write summaries
        # TODO: Think of a good way to write rec output summaries - rn, just using first output.
        self.summary_writer.add_summary(rec_summaries[0], global_step)

        ##
        # Save images
        ##

        if save_imgs:
            for pred_num in range(len(input_frames)):
                pred_dir = c.get_dir(os.path.join(
                    c.IMG_SAVE_DIR, 'Tests/Step_' + str(global_step), str(pred_num)))

                # save input images
                for frame_num in range(c.HIST_LEN):
                    img = input_frames[pred_num, :, :, (frame_num * 3):((frame_num + 1) * 3)]
                    imsave(os.path.join(pred_dir, 'input_' + str(frame_num) + '.png'), img)

                # save recursive outputs
                for rec_num in range(num_rec_out):
                    gen_img = rec_preds[rec_num][pred_num]
                    gt_img = gt_frames[pred_num, :, :, 3 * rec_num:3 * (rec_num + 1)]
                    imsave(os.path.join(pred_dir, 'gen_' + str(rec_num) + '.png'), gen_img)
                    imsave(os.path.join(pred_dir, 'gt_' + str(rec_num) + '.png'), gt_img)

        print '-' * 30

def combined_loss(gen_frames, gt_frames, d_preds, lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2):
    """
    Calculates the sum of the combined adversarial, lp and GDL losses in the given proportion. Used
    for training the generative model.

    Args:
    - gen_frames: A list of tensors of the generated frames at each scale.
    - gt_frames: A list of tensors of the ground truth frames at each scale.
    - d_preds: A list of tensors of the classifications made by the discriminator model at each
               scale.
    - lam_adv: The percentage of the adversarial loss to use in the combined loss.
    - lam_lp: The percentage of the lp loss to use in the combined loss.
    - lam_gdl: The percentage of the GDL loss to use in the combined loss.
    - l_num: 1 or 2 for l1 and l2 loss, respectively.
    - alpha: The power to which each gradient term is raised in GDL loss.

    Returns:
    - The combined adversarial, lp and GDL losses.
    """
    batch_size = gen_frames[0].size(0)  # variable batch size as a tensor

    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    if c.ADVERSARIAL:
        loss += lam_adv * adv_loss(d_preds, torch.ones([batch_size, 1], dtype=torch.float))

    return loss


def sharp_diff_error(gen_frames, gt_frames):
    """
    Computes the Sharpness Difference error between the generated images and the ground truth
    images.

    Args:
    - gen_frames: A tensor of shape [batch_size, 3, height, width]. The frames generated by the
                  generator model.
    - gt_frames: A tensor of shape [batch_size, 3, height, width]. The ground-truth frames for
                 each frame in gen_frames.

    Returns:
    - A scalar tensor. The Sharpness Difference error over each frame in the batch.
    """
    shape = gen_frames.size()
    num_pixels = torch.tensor(shape[2] * shape[3], dtype=torch.float)

    # gradient difference
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    # TODO: Could this be simplified with one filter [[-1, 2], [0, -1]]?
    pos = torch.tensor(np.identity(3), dtype=torch.float)
    neg = -1 * pos
    filter_x = torch.unsqueeze(torch.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = torch.stack([torch.unsqueeze(pos, 0), torch.unsqueeze(neg, 0)])  # [[1],[-1]]

    gen_dx = torch.abs(torch.nn.functional.conv2d(gen_frames, filter_x, padding=1))
    gen_dy = torch.abs(torch.nn.functional.conv2d(gen_frames, filter_y, padding=1))
    gt_dx = torch.abs(torch.nn.functional.conv2d(gt_frames, filter_x, padding=1))
    gt_dy = torch.abs(torch.nn.functional.conv2d(gt_frames, filter_y, padding=1))

    gen_grad_sum = gen_dx + gen_dy
    gt_grad_sum = gt_dx + gt_dy

    grad_diff = torch.abs(gt_grad_sum - gen_grad_sum)

    batch_errors = 10 * torch.log10(1 / ((1 / num_pixels) * torch.sum(grad_diff, dim=(2, 3))))
    return torch.mean(batch_errors)

def w(shape, stddev=0.01):
    """
    @return A weight layer with the given shape and standard deviation. Initialized with a
            truncated normal distribution.
    """
    return nn.Parameter(torch.randn(shape) * stddev)

def b(shape, const=0.1):
    """
    @return A bias layer with the given shape.
    """
    return nn.Parameter(torch.ones(shape) * const)

def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    num_pixels = gen_frames.size(1) * gen_frames.size(2) * gen_frames.size(3)
    square_diff = (gt_frames - gen_frames).pow(2)

    batch_errors = 10 * torch.log10(1 / ((1 / num_pixels) * square_diff.sum((1, 2, 3))))
    return batch_errors.mean()
