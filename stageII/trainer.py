from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
import json
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
from PIL import Image, ImageDraw, ImageFont
import pickle
from misc.utils import transform as image_transform


from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

        self.hr_image_shape = self.dataset.image_shape
        ratio = self.dataset.hr_lr_ratio
        self.lr_image_shape = [int(self.hr_image_shape[0] / ratio),
                               int(self.hr_image_shape[1] / ratio),
                               self.hr_image_shape[2]]
        print('hr_image_shape', self.hr_image_shape)
        print('lr_image_shape', self.lr_image_shape)

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.hr_images = tf.placeholder(
            tf.float32, [None] + self.hr_image_shape,
            name='real_hr_images')
        self.hr_wrong_images = tf.placeholder(
            tf.float32, [None] + self.hr_image_shape,
            name='wrong_hr_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [None] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )
        #
        self.images = tf.image.resize_bilinear(self.hr_images,
                                               self.lr_image_shape[:2])
        self.wrong_images = tf.image.resize_bilinear(self.hr_wrong_images,
                                                     self.lr_image_shape[:2])

    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        # Build conditioning augmentation structure for text embedding
        # under different variable_scope: 'g_net' and 'hr_g_net'
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0
        # TODO: play with the coefficient for KL
        return c, cfg.TRAIN.COEFF.KL * kl_loss

    def init_opt(self):
        self.build_placeholder()
        '''
        with pt.defaults_scope(phase=pt.Phase.train):
            ### Define low and high resolution discriminator
            d_loss = self.make_discriminator(self.images, self.wrong_images, self.embeddings, flag='lr')
            hr_d_loss = self.make_discriminator(self.hr_images, self.hr_wrong_images, self.embeddings, flag='hr')
            ## Define trainer for the discriminators, with provided learning rates
            self.discriminator_trainer = self.define_one_trainer(d_loss, self.discriminator_lr, 'd_')
            self.hr_discriminator_trainer = self.define_one_trainer(hr_d_loss, self.discriminator_lr, 'hr_d_')
            self.log_vars.append(("hr_d_learning_rate", self.discriminator_lr))
        '''
        '''
            # ####get output from G network####################################
            with tf.variable_scope("g_net"):
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([tf.shape(self.embeddings)[0], cfg.Z_DIM])
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                fake_images = self.model.get_generator(tf.concat(axis=1, values=[c, z]))

            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings,
                                    flag='lr')
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #### For hr_g and hr_d #########################################
            with tf.variable_scope("hr_g_net"):
                hr_c, hr_kl_loss = self.sample_encoded_context(self.embeddings)
                self.log_vars.append(("hist_hr_c", hr_c))
                hr_fake_images = self.model.hr_get_generator(fake_images, hr_c)
            # get losses
            hr_discriminator_loss, hr_generator_loss =\
                self.compute_losses(self.hr_images,
                                    self.hr_wrong_images,
                                    hr_fake_images,
                                    self.embeddings,
                                    flag='hr')
            hr_generator_loss += hr_kl_loss
            self.log_vars.append(("hr_g_loss", hr_generator_loss))
            self.log_vars.append(("hr_d_loss", hr_discriminator_loss))

            # #######define self.g_sum, self.d_sum,....########################
            self.prepare_trainer(discriminator_loss, generator_loss,
                                 hr_discriminator_loss, hr_generator_loss)
            self.define_summaries()
        '''
        ### collect summaries
        #self.define_summaries()
        with pt.defaults_scope(phase=pt.Phase.test):
            #self.sampler()
            self.critic()
            ### make accuracy evaluations
            self.eval_discriminator(self.images, self.wrong_images, self.embeddings, flag='lr')
            self.eval_discriminator(self.hr_images, self.hr_wrong_images, self.embeddings, flag='hr')
            #self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler(self):
        with tf.variable_scope("g_net", reuse=True):
            c, _ = self.sample_encoded_context(self.embeddings)
            z = tf.random_normal([tf.shape(self.embeddings)[0], cfg.Z_DIM])
            self.fake_images = self.model.get_generator(tf.concat(axis=1, values=[c, z]))
        with tf.variable_scope("hr_g_net", reuse=True):
            hr_c, _ = self.sample_encoded_context(self.embeddings)
            self.hr_fake_images =\
                self.model.hr_get_generator(self.fake_images, hr_c)

    def critic(self):
        self.critic_logits =\
            self.model.get_discriminator(self.images, self.embeddings)
        self.hr_critic_logits =\
            self.model.hr_get_discriminator(self.hr_images, self.embeddings)

    def make_discriminator(self, images, wrong_images, embeddings, flag='lr'):
        if flag == 'lr':
            real_logit =\
                self.model.get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.get_discriminator(wrong_images, embeddings)
        else:
            real_logit =\
                self.model.hr_get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.hr_get_discriminator(wrong_images, embeddings)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                    labels=tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit,
                                                    labels=tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        discriminator_loss = real_d_loss + wrong_d_loss
        if flag == 'lr':
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_wrong", wrong_d_loss))
            self.log_vars.append(("d_loss", discriminator_loss))
        else:
            self.log_vars.append(("hr_d_loss_real", real_d_loss))
            self.log_vars.append(("hr_d_loss_wrong", wrong_d_loss))
            self.log_vars.append(("hr_d_loss", discriminator_loss))

        return discriminator_loss

    def eval_discriminator(self, images, wrong_images, embeddings, flag='lr'):
        if flag == 'lr':
            real_logit =\
                self.model.get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.get_discriminator(wrong_images, embeddings)
        else:
            real_logit =\
                self.model.hr_get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.hr_get_discriminator(wrong_images, embeddings)

        real_preds = tf.cast(tf.greater(real_logit,0.5), tf.float32)
        real_preds = tf.reduce_mean(real_preds)
        wrong_preds = tf.cast(tf.less(wrong_logit,0.5), tf.float32)
        wrong_preds = tf.reduce_mean(wrong_preds)
        #_, real_acc = tf.metrics.accuracy(labels=tf.ones_like(real_preds), predictions=real_preds)
        #_, wrong_acc = tf.metrics.accuracy(labels=tf.zeros_like(wrong_preds), predictions=wrong_preds)
        #total_acc = real_acc + wrong_acc / 2.0
        if flag == 'lr':
            self.log_vars.append(("d_preds_real", real_preds))
            self.log_vars.append(("d_preds_wrong", wrong_preds))
        else:
            self.log_vars.append(("hr_d_preds_real", real_preds))
            self.log_vars.append(("hr_d_preds_wrong", wrong_preds))


        #return total_acc
        

    def compute_losses(self, images, wrong_images,
                       fake_images, embeddings, flag='lr'):
        if flag == 'lr':
            real_logit =\
                self.model.get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.get_discriminator(wrong_images, embeddings)
            fake_logit =\
                self.model.get_discriminator(fake_images, embeddings)
        else:
            real_logit =\
                self.model.hr_get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.hr_get_discriminator(wrong_images, embeddings)
            fake_logit =\
                self.model.hr_get_discriminator(fake_images, embeddings)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                    labels=tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit,
                                                    labels=tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                    labels=tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss =\
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        if flag == 'lr':
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_fake", fake_d_loss))
            if cfg.TRAIN.B_WRONG:
                self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            self.log_vars.append(("hr_d_loss_real", real_d_loss))
            self.log_vars.append(("hr_d_loss_fake", fake_d_loss))
            if cfg.TRAIN.B_WRONG:
                self.log_vars.append(("hr_d_loss_wrong", wrong_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                    labels=tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)
        if flag == 'lr':
            self.log_vars.append(("g_loss_fake", generator_loss))
        else:
            self.log_vars.append(("hr_g_loss_fake", generator_loss))

        return discriminator_loss, generator_loss

    def define_one_trainer(self, loss, learning_rate, key_word):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()
        tarin_vars = [var for var in all_vars if
                      var.name.startswith(key_word)]

        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        trainer = pt.apply_optimizer(opt, losses=[loss], var_list=tarin_vars)
        return trainer

    def prepare_trainer(self, discriminator_loss, generator_loss,
                        hr_discriminator_loss, hr_generator_loss):
        ft_lr_retio = cfg.TRAIN.FT_LR_RETIO
        self.discriminator_trainer =\
            self.define_one_trainer(discriminator_loss,
                                    self.discriminator_lr * ft_lr_retio,
                                    'd_')
        self.generator_trainer =\
            self.define_one_trainer(generator_loss,
                                    self.generator_lr * ft_lr_retio,
                                    'g_')
        self.hr_discriminator_trainer =\
            self.define_one_trainer(hr_discriminator_loss,
                                    self.discriminator_lr,
                                    'hr_d_')
        self.hr_generator_trainer =\
            self.define_one_trainer(hr_generator_loss,
                                    self.generator_lr,
                                    'hr_g_')

        self.ft_generator_trainer = \
            self.define_one_trainer(hr_generator_loss,
                                    self.generator_lr * cfg.TRAIN.FT_LR_RETIO,
                                    'g_')

        self.log_vars.append(("hr_d_learning_rate", self.discriminator_lr))
        self.log_vars.append(("hr_g_learning_rate", self.generator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hr_g': [], 'hr_d': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hr_g'):
                all_sum['hr_g'].append(tf.summary.scalar(k, v))
            elif k.startswith('hr_d'):
                all_sum['hr_d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.summary.histogram(k, v))

        #self.g_sum = tf.summary.merge(all_sum['g'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        #self.hr_g_sum = tf.summary.merge(all_sum['hr_g'])
        self.hr_d_sum = tf.summary.merge(all_sum['hr_d'])
        #self.hist_sum = tf.summary.merge(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(axis=1, values=row_img))
        imgs = tf.expand_dims(tf.concat(axis=0, values=stacked_img), 0)
        current_img_summary = tf.summary.image(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train =\
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test =\
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat(axis=0, values=[superimage_train, superimage_test])
        self.image_summary = tf.summary.merge([fake_sum_train, fake_sum_test])

        hr_fake_sum_train, hr_superimage_train =\
            self.visualize_one_superimage(self.hr_fake_images[:n * n],
                                          self.hr_images[:n * n, :, :, :],
                                          n, "hr_train")
        hr_fake_sum_test, hr_superimage_test =\
            self.visualize_one_superimage(self.hr_fake_images[n * n:2 * n * n],
                                          self.hr_images[n * n:2 * n * n],
                                          n, "hr_test")
        self.hr_superimages =\
            tf.concat(axis=0, values=[hr_superimage_train, hr_superimage_test])
        self.hr_image_summary =\
            tf.summary.merge([hr_fake_sum_train, hr_fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ =\
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        feed_out = [self.superimages, self.image_summary,
                    self.hr_superimages, self.hr_image_summary]
        feed_dict = {self.hr_images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary, hr_gen_samples, hr_img_summary =\
            sess.run(feed_out, feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/lr_fake_train.jpg' %
                          (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/lr_fake_test.jpg' %
                          (self.log_dir), gen_samples[1])
        #
        scipy.misc.imsave('%s/hr_fake_train.jpg' %
                          (self.log_dir), hr_gen_samples[0])
        scipy.misc.imsave('%s/hr_fake_test.jpg' %
                          (self.log_dir), hr_gen_samples[1])

        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary, hr_img_summary

    def build_model(self, sess):
        self.init_opt()

        sess.run(tf.global_variables_initializer())
        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            all_vars = tf.trainable_variables()
            # all_vars = tf.all_variables()
            restore_vars = []
            for var in all_vars:
                if var.name.startswith('g_') or var.name.startswith('d_'):
                    restore_vars.append(var)
                    # print(var.name)
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train_one_step(self, generator_lr,
                       discriminator_lr,
                       counter, summary_writer, log_vars, sess):
        # training d
        hr_images, hr_wrong_images, embeddings, _, _ =\
            self.dataset.train.next_batch(self.batch_size,
                                          cfg.TRAIN.NUM_EMBEDDING)
        feed_dict = {self.hr_images: hr_images,
                     self.hr_wrong_images: hr_wrong_images,
                     self.embeddings: embeddings,
                     self.generator_lr: generator_lr,
                     self.discriminator_lr: discriminator_lr
                     }
        if cfg.TRAIN.FINETUNE_LR:
            # train d1
            feed_out_d = [self.hr_discriminator_trainer,
                          self.hr_d_sum,
                          log_vars,
                          self.hist_sum]
            ret_list = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(ret_list[1], counter)
            log_vals = ret_list[2]
            summary_writer.add_summary(ret_list[3], counter)
            # train g1 and finetune g0 with the loss of g1
            feed_out_g = [self.hr_generator_trainer,
                          self.ft_generator_trainer,
                          self.hr_g_sum]
            _, _, hr_g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(hr_g_sum, counter)
            # finetune d0 with the loss of d0
            feed_out_d = [self.discriminator_trainer, self.d_sum]
            _, d_sum = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(d_sum, counter)
            # finetune g0 with the loss of g0
            feed_out_g = [self.generator_trainer, self.g_sum]
            _, g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(g_sum, counter)
        else:
            # train d1
            feed_out_d = [self.hr_discriminator_trainer,
                          self.hr_d_sum,
                          log_vars,
                          self.hist_sum]
            ret_list = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(ret_list[1], counter)
            log_vals = ret_list[2]
            summary_writer.add_summary(ret_list[3], counter)
            # train g1
            feed_out_g = [self.hr_generator_trainer,
                          self.hr_g_sum]
            _, hr_g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(hr_g_sum, counter)

        return log_vals

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.global_variables(),
                                       keep_checkpoint_every_n_hours=5)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)

                if cfg.TRAIN.FINETUNE_LR:
                    keys = ["hr_d_loss", "hr_g_loss", "d_loss", "g_loss"]
                else:
                    keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(np.ceil(number_example * 1.0 / self.batch_size))
                # int((counter + lr_decay_step/2) / lr_decay_step)
                decay_start = cfg.TRAIN.PRETRAINED_EPOCH
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch > decay_start:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        log_vals = self.train_one_step(generator_lr,
                                                       discriminator_lr,
                                                       counter, summary_writer,
                                                       log_vars, sess)
                        all_log_vals.append(log_vals)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    img_summary, img_summary2 =\
                        self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_summary, counter)
                    summary_writer.add_summary(img_summary2, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

    def drawCaption(self, img, caption):
        img_txt = Image.fromarray(img)
        # get a font
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
        # get a drawing context
        d = ImageDraw.Draw(img_txt)

        # draw text, half opacity
        d.text((10, 256), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
        d.text((10, 512), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))
        if img.shape[0] > 832:
            d.text((10, 832), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
            d.text((10, 1088), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))

        idx = caption.find(' ', 60)
        if idx == -1:
            d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
        else:
            cap1 = caption[:idx]
            cap2 = caption[idx+1:]
            d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
            d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

        return img_txt

    def save_super_images(self, images, sample_batchs, hr_sample_batchs,
                          savenames, captions_batchs,
                          sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        # Up to 16 samples for each text embedding/sentence
        numSamples = len(sample_batchs)
        for j in range(len(savenames)):
            s_tmp = '%s-1real-%dsamples/%s/%s' %\
                (save_dir, numSamples, subset, savenames[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            # First row with up to 8 samples
            real_img = (images[j] + 1.0) * 127.5
            img_shape = real_img.shape
            padding0 = np.zeros(img_shape)
            padding = np.zeros((img_shape[0], 20, 3))

            row1 = [padding0, real_img, padding]
            row2 = [padding0, real_img, padding]
            for i in range(np.minimum(8, numSamples)):
                lr_img = sample_batchs[i][j]
                hr_img = hr_sample_batchs[i][j]
                hr_img = (hr_img + 1.0) * 127.5
                re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                row1.append(re_sample)
                row2.append(hr_img)
            row1 = np.concatenate(row1, axis=1)
            row2 = np.concatenate(row2, axis=1)
            superimage = np.concatenate([row1, row2], axis=0)

            # Second 8 samples with up to 8 samples
            if len(sample_batchs) > 8:
                row1 = [padding0, real_img, padding]
                row2 = [padding0, real_img, padding]
                for i in range(8, len(sample_batchs)):
                    lr_img = sample_batchs[i][j]
                    hr_img = hr_sample_batchs[i][j]
                    hr_img = (hr_img + 1.0) * 127.5
                    re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                    row1.append(re_sample)
                    row2.append(hr_img)
                row1 = np.concatenate(row1, axis=1)
                row2 = np.concatenate(row2, axis=1)
                super_row = np.concatenate([row1, row2], axis=0)
                superimage2 = np.zeros_like(superimage)
                superimage2[:super_row.shape[0],
                            :super_row.shape[1],
                            :super_row.shape[2]] = super_row
                mid_padding = np.zeros((64, superimage.shape[1], 3))
                superimage = np.concatenate([superimage, mid_padding,
                                             superimage2], axis=0)

            top_padding = np.zeros((128, superimage.shape[1], 3))
            superimage =\
                np.concatenate([top_padding, superimage], axis=0)

            captions = captions_batchs[j][sentenceID]
            fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
            superimage = self.drawCaption(np.uint8(superimage), captions)
            scipy.misc.imsave(fullpath, superimage)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images, embeddings_batchs, savenames, captions_batchs =\
                dataset.next_batch_test(self.batch_size, start, 1)

            print('count = ', count, 'start = ', start)
            # the i-th sentence/caption
            for i in range(len(embeddings_batchs)):
                samples_batchs = []
                hr_samples_batchs = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                numSamples = np.minimum(16, cfg.TRAIN.NUM_COPY)
                for j in range(numSamples):
                    hr_samples, samples =\
                        sess.run([self.hr_fake_images, self.fake_images],
                                 {self.embeddings: embeddings_batchs[i]})
                    samples_batchs.append(samples)
                    hr_samples_batchs.append(hr_samples)
                self.save_super_images(images, samples_batchs,
                                       hr_samples_batchs,
                                       savenames, captions_batchs,
                                       i, save_dir, subset)

            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")

    def zero_shot_eval(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    #sess.run(tf.global_variables_initializer())
                    print(">>>Zero shot evaluation started")
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.global_variables())
                    with open('zsl_logs/koy_birds_model_dict.pickle', 'wb') as fs:
                        gvn = [gv.name for gv in tf.global_variables()]
                        gvs = [gv.shape for gv in tf.global_variables()]
                        pickle.dump(dict(zip(gvn, gvs)), fs)
                    #saver.save(sess, 'zsl_logs/koy_birds_model.ckpt')
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.zs_eval_one_dataset(sess, self.dataset.test,
                                          'zsl_logs/logs', subset='test')
                else:
                    print("Input a valid model path.")

    def zs_eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        '''for each image, go through all embeddings'''
        ### read images, class_ids and embeddings(10 per image) from dataset
        ### flat embeddings
        os.system('mkdir -p '+ save_dir)
        embeddings = dataset._embeddings
        #embeddings_flat = embeddings.reshape([embeddings.shape[0]*embeddings.shape[1], embeddings.shape[2]])
        class_ids = dataset._class_id.astype(int)
        #class_ids_extend = np.array([class_ids[cid//embeddings.shape[1]] for cid in range(embeddings_flat.shape[0])])
        filenames = dataset._filenames
        images = dataset._images

        ## list of selected embeddings ids
        sent_select = list()
        class_dict = defaultdict(list)
        for i, c in enumerate(class_ids):
            class_dict[c].append(i)
        for c, id_list in class_dict.items():
            assert class_sent_count < len(id_list), ">>> number of class sentences more that total sentences in that class"
            choice = np.random.choice(id_list, size=cfg.ZEROSHOT.SENT_PER_CLASS, replace=False)
            sent_select += choice

        ## select the chosen ids from embeddings, class_ids, and filenames
        sel_embeddings = embeddings[sent_select,...]
        embeddings_flat = sel_embeddings.reshape([sel_embeddings.shape[0]*sel_embeddings.shape[1], sel_embeddings.shape[2]])
        sel_class_ids = class_ids[sent_select,...]
        class_ids_extend = np.array([sel_class_ids[cid//sel_embeddings.shape[1]] for cid in range(embeddings_flat.shape[0])])
        sel_filenames = filenames[sent_select,...]

        print('>>> embeddings shape:')
        print(sel_embeddings.shape)
        
        print('>>> flat embeddings shape:')
        print(embeddings_flat.shape)
        
        ### for each image, pair with all embeddings in several batches
        batch_count = 0
        for batch_start in range(0, embeddings_flat.shape[0], self.batch_size):
            widgets = ["sent_batch #%d|" % (batch_start*100//embeddings_flat.shape[0]), Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=images.shape[0], widgets=widgets)
            pbar.start()
            
            batch_end = batch_start + self.batch_size
            batch_embeddings = embeddings_flat[batch_start:batch_end]
            batch_sent_cids = class_ids_extend[batch_start:batch_end]
            this_batch_size = batch_embeddings.shape[0]
            ## read filenames corresponding to current batch embeddings
            bid = batch_start
            batch_caps = list()
            while bid < batch_end:
                fn = sel_filenames[bid // embeddings.shape[1]]
                cap_path = '%s/text_c10/%s.txt' % (dataset.workdir, fn)
                with open(cap_path, 'r') as fs:
                    for lid, cap in enumerate(fs):
                        if lid == bid % embeddings.shape[1]:
                            batch_caps.append(cap)
                            bid += 1
            ## for each image, batch with current batch embeddings
            im_count = 0
            for im, c in zip(images, class_ids):
                ## tile one image as many as embeddings in the batch
                pbar.update(im_count)
                im_trans = image_transform(im * (2. / 255) - 1, self.hr_image_shape[0], is_crop=False, bbox=None)
                batch_images = np.tile(im_trans, (this_batch_size,1,1,1))
                batch_im_cids = np.tile([c],this_batch_size)
                #print(">>> current_batch_size:", batch_images.shape[0])
                hr_critic_logits, critic_logits =\
                    sess.run([self.hr_critic_logits, self.critic_logits],
                             {self.embeddings: batch_embeddings, self.hr_images: batch_images})
                hr_critic_logits = hr_critic_logits.flatten().astype(float)
                critic_logits = critic_logits.flatten().astype(float)
                batch_preds = [{'im_name': filenames[im_count], 'im_cid': imc, 'sent_cid':sc, 'hr_prob': hr_ds, 'prob': ds, 'parses': p}\
                                for imc, sc, hr_ds, ds, p in\
                                zip(batch_im_cids, batch_sent_cids, hr_critic_logits, critic_logits, batch_caps)]
                if len(batch_preds) == 0:
                    print('start:%d --- end:%d' %(batch_start, batch_end))
                    print('filenames: %d' % len(filenames[batch_start:batch_end]))
                    print(batch_im_cids.shape)
                    print(batch_sent_cids.shape)
                    print(hr_critic_logits.shape)
                    print(critic_logits.shape)
                    print('batch caps: %d' % len(batch_caps))
                with open('%s/batch_%d.json' % (save_dir, batch_count), 'w+') as fj:
                    json.dump(batch_preds, fj, indent=4)
                batch_count += 1
                im_count += 1

    def train_classifier_one_step(self, discriminator_lr, counter, summary_writer, log_vars, sess):
        # training high resolution discriminator
        hr_images, hr_wrong_images, embeddings, _, _ =\
            self.dataset.train.next_batch(self.batch_size,
                                          cfg.TRAIN.NUM_EMBEDDING)
        feed_dict = {self.hr_images: hr_images,
                     self.hr_wrong_images: hr_wrong_images,
                     self.embeddings: embeddings,
                     self.discriminator_lr: discriminator_lr
                     }
        feed_out_d = [self.discriminator_trainer,
                        self.hr_discriminator_trainer,
                        self.d_sum,
                        self.hr_d_sum,
                        log_vars]
        ret_list = sess.run(feed_out_d, feed_dict)
        summary_writer.add_summary(ret_list[2], counter)
        summary_writer.add_summary(ret_list[3], counter)
        log_vals = ret_list[4]
        return log_vals

    def eval_classifier_one_step(self, discriminator_lr, sess):
        # training high resolution discriminator
        hr_images, hr_wrong_images, embeddings, _, _ =\
            self.dataset.test.next_batch(self.batch_size,
                                          cfg.TRAIN.NUM_EMBEDDING)
        feed_dict = {self.hr_images: hr_images,
                     self.hr_wrong_images: hr_wrong_images,
                     self.embeddings: embeddings,
                     self.discriminator_lr: discriminator_lr
                     }

        for k, v in self.log_vars:
            if k == 'd_preds_real':
                d_preds_real = v
            elif k == 'd_preds_wrong':
                d_preds_wrong = v
            if k == 'hr_d_preds_real':
                hr_d_preds_real = v
            elif k == 'hr_d_preds_wrong':
                hr_d_preds_wrong = v

        feed_out = [d_preds_real, d_preds_wrong, hr_d_preds_real, hr_d_preds_wrong]
        ret_list = sess.run(feed_out, feed_dict)
    
        return ret_list

    def train_classifier(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.global_variables(),
                                       keep_checkpoint_every_n_hours=5)

                summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
                keys = ["hr_d_loss", "d_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                number_example_eval = self.dataset.test._num_examples
                updates_per_epoch = int(np.ceil(number_example * 1.0 / self.batch_size))
                updates_per_epoch_eval = int(np.ceil(number_example_eval * 1.0 / self.batch_size))
                # int((counter + lr_decay_step/2) / lr_decay_step)
                decay_start = cfg.TRAIN.PRETRAINED_EPOCH
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    print('>>> TRAIN at epoch: %d' % epoch)
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch > decay_start:
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        log_vals = self.train_classifier_one_step(discriminator_lr, counter, summary_writer, log_vars, sess)
                        all_log_vals.append(log_vals)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    print('>>> EVALUATE at epoch: %d' % epoch)
                    widgets = ["eval_epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch_eval,
                                       widgets=widgets)
                    pbar.start()
                    ### evaluate on test
                    acc_list = list()
                    for i in range(updates_per_epoch_eval):
                        pbar.update(i)
                        acc = self.eval_classifier_one_step(discriminator_lr, sess)
                        acc_list.append(acc)
                    acc_mat = np.array(acc_list)
                    d_acc, dw_acc, hr_d_acc, hr_dw_acc = np.mean(acc_mat, axis=0)
                    total_acc = (d_acc + dw_acc) / 2.0
                    hr_total_acc = (hr_d_acc + hr_dw_acc) / 2.0
                    acc_sum = tf.Summary(value=[
                        tf.Summary.Value(tag="d_acc_real", simple_value=d_acc), 
                        tf.Summary.Value(tag="d_acc_wrong", simple_value=dw_acc),
                        tf.Summary.Value(tag="d_acc_total", simple_value=total_acc),
                        tf.Summary.Value(tag="hr_d_acc_real", simple_value=hr_d_acc),
                        tf.Summary.Value(tag="hr_d_acc_wrong", simple_value=hr_dw_acc),
                        tf.Summary.Value(tag="hr_d_acc_total", simple_value=hr_total_acc)])

                    summary_writer.add_summary(acc_sum, epoch)
                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)
                    dic_logs['total_acc'] = total_acc
                    dic_logs['hr_total_acc'] = hr_total_acc

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

