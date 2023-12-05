import os
import torch
from torch import Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms
import string
import pandas as pd
from random import randrange
from typing import Tuple, List, Callable, Iterable
from torch import types as pt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from cocoDataset import COCO_DS
from pathlib import Path
import logging

__COCO_IMG_PATH = Path("caption", "data", "val_004_image")

__COCO_ANN_PATH = Path("caption", "data", "annotate")

__TRAIN_PATH = {'root': __COCO_IMG_PATH.joinpath("train"),
                'annFile': __COCO_ANN_PATH.joinpath("train_annotate.csv")
                }
__VAL_PATH = {'root': __COCO_IMG_PATH.joinpath("val"),
              'annFile': __COCO_ANN_PATH.joinpath("val_annotate.csv")
              }

__UNK_TOKEN = 'UNK'
__PAD_TOKEN = 'PAD'
__EOS_TOKEN = 'EOS'

__normalize = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}

def unpack_sequence(packed_sequence: List[Tensor], lengths:List[int])->List[Tensor]:
    """Unpacks a packed sequence by lenghts. 

    Args:
        packed_sequence (List[Tensor]): Packed sequence of Indexes from rnn.packed_sequence function
        lengths (List[int]): lengths of sequence

    Returns:
        List[Tensor]: List of Long tensor of indexes
    """
    head = 0
    unpacked_sequence = [torch.zeros(l).int() for l in lengths]
    batch_size = len(lengths)
    indexes = [0 for l in lengths]
    while head < len(packed_sequence):
        for b in range(batch_size):
            if indexes[b]== lengths[b]:
                continue
            unpacked_sequence[b][indexes[b]] = packed_sequence[head]
            head+=1
            indexes[b]+=1
    return unpacked_sequence



def simple_tokenize(s:str)-> List[str]:
    """Tokenizes string by lower case, removing punctiations and removing spaces

    Args:
        s (str): String

    Returns:
        List[str]: Token of words
    """
    translator = str.maketrans('', '', string.punctuation)
    return s.lower().translate(translator).strip().split()

def build_vocab(annFile:str = __TRAIN_PATH['annFile'], num_words:int = 10000) -> List[str]:
    """Generate vocab of indexes for all tokens in file

    Args:
        annFile (str, optional): _description_. Defaults to __TRAIN_PATH['annFile']:str.
        num_words (int, optional): _description_. Defaults to 10000.

    Returns:
        List[str]: _description_
    """
    sentence_list = pd.read_csv(annFile).caption.tolist()
    from collections import Counter
    count_tok = Counter()
    for sentence in sentence_list:
        tok = simple_tokenize(sentence)
        count_tok+=Counter(tok)
    
    cw = sorted([(f,w) for w,f in count_tok.items() if f>1])[:num_words]
    vocab = [w for _,w in cw]
    vocab = [__PAD_TOKEN] + vocab + [__UNK_TOKEN, __EOS_TOKEN]
    return vocab

def get_target(vocab:List[str], rnd_caption = True)-> Callable:
    """Generates captions function that randomizes a caption from list of target captions. 
    Job of function is to take captions and convert to Tokens of indexes

    Args:
        vocab (List[str]): _description_
        rnd_caption (bool, optional): _description_. Defaults to True.

    Returns:
        Callable: _description_
    """
    word2Idx = {w: i for i, w in enumerate(vocab)}
    def get_caption(captions):
        idx = 0
        if rnd_caption:
            idx = randrange(len(captions))
        caption = simple_tokenize(captions[idx])
        return torch.Tensor([word2Idx.get(c, word2Idx[__UNK_TOKEN]) for c in caption])
    return get_caption

def get_coco_data(vocab:List[str], train:bool = True, img_size:int = 224, scale_size:int = 256, normalize: Dict[str:List[float]] = __normalize) -> Tuple[Iterable, List[str]]:
    """Get COCO dataset by performing Image transformation and caption transformation

    Args:
        vocab (List[str]): Vocab list 
        train (bool, optional): Train/Val. Defaults to True.
        img_size (int, optional): _description_. Defaults to 224.
        scale_size (int, optional): _description_. Defaults to 256.
        normalize (_type_, optional): Normalize image. Defaults to __normalize.

    Returns:
        Tuple[Iterable, List[str]]: Dataset, vocab
    """
    
    if train:
        root, annFile = __TRAIN_PATH['root'], __TRAIN_PATH['annFile']
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(scale_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees = (90,90)),
            transforms.RandomRotation(degrees = (180,180)),
            transforms.RandomRotation(degrees = (270,270)),
            transforms.ToTensor(),
            
            transforms.Normalize(**normalize)
        ])
    else:
        root, annFile = __VAL_PATH['root'], __VAL_PATH['annFile']
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(scale_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
        
    ds = COCO_DS(root, annFile, img_transform, get_target(vocab, rnd_caption=train))
    return ds, vocab

def create_batches(vocab:List[str], max_caption_len:int = 50) -> Callable:
    """Generate batches by adding padding to all the sequence in batch and arraning in Tensors of same sequence.
    Clips words > max_caption_len and adds EOS in the end

    Args:
        vocab (List[str]): vocab list
        max_caption_len (int, optional): _description_. Defaults to 50.

    Raises:
        e: _description_

    Returns:
        Callable: Collate functions
    """
    padding = vocab.index(__PAD_TOKEN)
    eos = vocab.index(__EOS_TOKEN)

    def collate_fn(img_cap)-> Tuple[Iterable, Tuple[Iterable, List[int]]]:
        """
        Sorts in reverse of caption len.
        Define batch_len = min(max_caption_len, max(length))
        Adds padding on caption sequence if len(caption) < batch_len
        Generate image tensor by stacking imgs
        """
        try:
            img_cap.sort(key = lambda x: len(x[1]), reverse = True)
        except Exception as e:
            logging.exception(e)
            print(e)
            raise e
        imgs, caps = zip(*img_cap)
        imgs = torch.stack(imgs)
        # Define batch len
        lengths = [min(len(cap) + 1, max_caption_len) for cap in caps]
        batch_len = max(lengths)
        capTensor = torch.LongTensor(batch_len, len(caps)).fill_(padding)
        # Copy captions to tensor
        for i,c in enumerate(caps):
            end_cap_i = lengths[i] - 1
            if  end_cap_i < batch_len:
                capTensor[end_cap_i, i] = eos
            capTensor[:end_cap_i,i].copy_(c[:end_cap_i])
        return (imgs, (capTensor, lengths))
    return collate_fn
            


def get_iterator(data, batch_size=32, max_length=30, shuffle=True, num_workers=4, pin_memory=True):
    cap, vocab = data
    return torch.utils.data.DataLoader(cap, batch_size=batch_size, shuffle=shuffle, 
                                  num_workers = num_workers, pin_memory= pin_memory, 
                                  collate_fn = create_batches(vocab, max_length), drop_last=True
                                  )


        


