from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dateutil
import dateutil.tz
import datetime
import argparse
import pprint
import os

from misc.datasets import TextDataset
from stageII.model import CondGAN
from stageII.trainer import CondGANTrainer
from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir,  cfg.EMBEDDING_TYPE, 4)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train)
        ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = CondGAN(
        lr_imsize=int(dataset.image_shape[0] / dataset.hr_lr_ratio),
        hr_lr_ratio=dataset.hr_lr_ratio
    )

    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )

    if cfg.TRAIN.FLAG:
        #algo.train()
        algo.train_classifier()
        algo.batch_size = 100
        alog.zero_shot_eval()

    elif cfg.ZEROSHOT.FLAG:
        '''
        For every input image in test dataset, calculate conditional probability given
        every sentence of all classes.
        '''
        algo.zero_shot_eval()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        #algo.evaluate()
        algo.eval_classifier()
