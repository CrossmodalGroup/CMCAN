"""Data provider"""

import torch
import torch.utils.data as data

import os
import numpy as np

import json


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # load the raw captions
        self.captions = []
        with open(loc + '%s_precaps_stan.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        self.adjs = np.load(loc + '%s_ims_dir_selfadj4.npy' % data_split)

        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        with open(loc + '%s_caps.json' % data_split) as f:
            self.depends = json.load(f)

        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = torch.Tensor(self.images[img_id])
        adj = torch.Tensor(self.adjs[img_id])
        caption = self.captions[index]
        depend = self.depends[index]

        # convert caption (string) to word ids.
        cap = []
        cap.extend(caption.decode('utf-8').split(','))
        #cap = [int(item) for _,item in enumerate(cap)]
        cap = list(map(int, cap))

        caption = torch.Tensor(cap)

        return image, caption, adj, depend, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, adjs, depends, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)
    # Merge adjacent matrix
    adjs = torch.stack(adjs, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, adjs, depends, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader
