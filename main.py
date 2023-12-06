import argparse
import os
import time
from datetime import datetime
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
from torch.nn.utils.clip_grad import clip_grad_norm
from data_utils import build_vocab, get_coco_data, get_iterator
from utils import setup_logging, adjust_optimizer, AverageMeter, select_optimizer
from model import CaptionModel
from torchvision.models import resnet
import torch

from pathlib import Path
from PIL import Image
import pandas as pd


model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='COCO caption genration training')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='caption/results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--cnn', '-a', metavar='CNN', default='resnet50',
                    choices=model_names,
                    help='cnn feature extraction architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='size of word embedding used')
parser.add_argument('--rnn_size', default=512, type=int,
                    help='size of rnn hidden layer')
parser.add_argument('--num_layers', default=2, type=int,
                    help='number of rnn layers to use')
parser.add_argument('--max_length', default=30, type=int,
                    help='maximum time length to feed')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-eb', '--eval_batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--grad_clip', default=5., type=float,
                    help='gradient max norm')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', '--learning_rate_decay', default=0.8, type=float,
                    metavar='LR', help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--share_weights', default=False, type=bool,
                    help='share embedder and classifier weights')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')

def arg_boilerplate():
    global args
    args = parser.parse_args()
    if args.save or '' == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    os.makedirs(save_path, exist_ok=True)

    setup_logging(os.path.join(save_path, 'log.txt'))

    logging.debug("run arguments: %s", args)
    logging.info("using pretrained cnn %s", args.cnn)




def main():
    global args, vocab
    arg_boilerplate()
    save_path = os.path.join(args.results_dir, args.save)
    cnn = resnet.__dict__[args.cnn](pretrained=True)
    checkpoint_file = os.path.join(save_path, 'checkpoint_epoch_%s.pth.tar')
    vocab = build_vocab() 
    model = CaptionModel(cnn, vocab,
                         embed_size=args.embedding_size,
                         rnn_size=args.rnn_size,
                         num_layers=args.num_layers,
                         share_embed_wts=args.share_weights)

    train_data = get_iterator(get_coco_data(vocab, train=True),
                              batch_size=args.batch_size,
                              max_length=args.max_length,
                              shuffle=True,
                              num_workers=args.workers)
    val_data = get_iterator(get_coco_data(vocab, train=False),
                            batch_size=args.eval_batch_size,
                            max_length=args.max_length,
                            shuffle=False,
                            num_workers=args.workers)

    if 'cuda' in args.type:
        cudnn.benchmark = True
        model.cuda()

    optimizer = select_optimizer(
        args.optimizer, params=model.parameters(), lr=args.lr)
    regime = lambda e: {'lr': args.lr * (args.lr_decay ** e),
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay}
    def forward(model, data, training=True, optimizer=None):
        use_cuda = 'cuda' in args.type
        loss = nn.CrossEntropyLoss()
        perplexity = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if training:
            model.train()
        else:
            model.eval()

        end = time.time()
        for i, (imgs, (captions, lengths)) in enumerate(data):
            data_time.update(time.time() - end)
            if use_cuda:
                imgs = imgs.cuda()
                captions = captions.cuda()
            imgs = Variable(imgs, volatile=not training)
            captions = Variable(captions, volatile=not training)
            input_captions = captions[:-1]

            pred = model(imgs, input_captions, lengths)
            err = loss(pred.reshape(-1, len(vocab)), captions.reshape(-1))
            perplexity.update(math.exp(err.item()))

            if training:
                optimizer.zero_grad()
                err.backward()
                clip_grad_norm(model.lstm.parameters(), args.grad_clip)
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Perplexity {perp.val:.4f} ({perp.avg:.4f})'.format(
                                 epoch, i, len(data),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time, perp=perplexity))

        return perplexity.avg

    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(
            optimizer, epoch, regime)
        # Train
        train_perp = forward(
            model, train_data, training=True, optimizer=optimizer)
        # Evaluate
        val_perp = forward(model, val_data, training=False)

        logging.info('\n Epoch: {0}\t'
                     'Training Perplexity {train_perp:.4f} \t'
                     'Validation Perplexity {val_perp:.4f} \n'
                     .format(epoch + 1, train_perp=train_perp, val_perp=val_perp))
        print(f"Saving in {checkpoint_file}")
        with torch.no_grad():
            show_and_tell(model)
        model.save_checkpoint(checkpoint_file % (epoch + 1))
    return model

"""
Display image and see the caption performance
"""
__COCO_IMG_PATH = Path( "caption", "data", "val_004_image")

__COCO_ANN_PATH = Path("caption", "data", "annotate")

__TRAIN_PATH = {'root': __COCO_IMG_PATH.joinpath("train"),
                'annFile': __COCO_ANN_PATH.joinpath("train_annotate.csv")
                }
ann_Val = pd.read_csv(__TRAIN_PATH['annFile']).groupby(['image_id', 'file_name']).apply(lambda x: x.caption.tolist()).reset_index(name='caption').head()
ann_Val.head()


def show_and_tell(model, filename = None, beam_size=1):
    if filename is None:
        row = ann_Val.sample()
        filename = row['file_name'].item()
    rows = ann_Val.query(f"file_name == @filename")
    print("Caption: ", rows['caption'].tolist())
    p = __TRAIN_PATH['root'].joinpath(filename)
    filename = str(p)
    with torch.no_grad():
        img = Image.open(filename, 'r')
        model.eval()
        captions = model.sample(img)
        print("Predicted captions: ", captions)

if __name__ == '__main__':
    model = main()
    while True:
        show_and_tell(model)
        print("done")


    
