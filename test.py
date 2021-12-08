#!/usr/bin/env python

from collections import defaultdict

import argparse
import importlib

# torchim:
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset

import numpy as np


# data:
import data
from data.collate import collate_fn, gpu_collate, no_pad_collate
from data.transforms import (
        Compose, AddLengths, AudioSqueeze, TextPreprocess,
        MaskSpectrogram, ToNumpy, BPEtexts, MelSpectrogram,
        ToGpu, Pad, NormalizedMelSpectrogram
)

# model:
from model import configs as quartznet_configs
from model.quartznet import QuartzNet

# utils:
import yaml
from easydict import EasyDict as edict
from utils import fix_seeds, remove_from_dict, prepare_bpe
import wandb
from decoder import GreedyDecoder, BeamCTCDecoder
from  VisdomUtils import VisdomLinePlotter

#plotter = VisdomLinePlotter(env_name='experiment1')
# TODO: wrap to trainer class
def train(config):
    fix_seeds(seed=config.train.get('seed', 42))
    dataset_module = importlib.import_module(f'.{config.dataset.name}', data.__name__)
    bpe = prepare_bpe(config)

    transforms_test = Compose([
            TextPreprocess(),
            ToNumpy(),
            BPEtexts(bpe=bpe),
            AudioSqueeze()
    ])

    batch_transforms_test = Compose([
            ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
            NormalizedMelSpectrogram(
                sample_rate=config.dataset.get('sample_rate', 16000), # for LJspeech
                n_mels=config.model.feat_in,
                normalize=config.dataset.get('normalize', None)
            ).to('cuda' if torch.cuda.is_available() else 'cpu'),
            AddLengths(),
            Pad()
    ])

    # load datasets
    test_dataset = dataset_module.get_dataset(config, transforms=transforms_test, part='test')

    test_dataloader = DataLoader(test_dataset, num_workers=0,
                batch_size=1, collate_fn=no_pad_collate)

    model = QuartzNet(
        model_config=getattr(quartznet_configs, config.model.name, '_quartznet5x5_config'),
        **remove_from_dict(config.model, ['name'])
    )
    model.load_state_dict(torch.load("/home/silviu/Desktop/projects/ASR-main/checkpoints/model_496_0.4398071173957665.pth"))
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    # criterion = nn.CTCLoss(blank=config.model.vocab_size)
    decoder = GreedyDecoder(bpe=bpe)



    # validate:
    model.eval()
    val_stats = defaultdict(list)
    val_loss = 0
    file2 = open(r"/home/silviu/Desktop/projects/ASR-main/data/test.txt", "w+")
    for batch_idx, batch in enumerate(test_dataloader):
        batch = batch_transforms_test(batch)
        with torch.no_grad():
            logits = model(batch['audio'])
            output_length = torch.ceil(batch['input_lengths'].float() / model.stride).int()
            loss = criterion(logits.permute(2, 0, 1).log_softmax(dim=2), batch['text'], output_length, batch['target_lengths'])
            val_loss += loss.item()

        target_strings = decoder.convert_to_strings(batch['text'])
        decoded_output = decoder.decode(logits.permute(0, 2, 1).softmax(dim=2))
        print("Target" + str( batch_idx))
        print(target_strings)
        print("Decode" + str( batch_idx))
        print(decoded_output)
        # file2.write(target_strings)
        # file2.write(decoded_output)
        wer = np.mean([decoder.wer(true, pred) for true, pred in zip(target_strings, decoded_output)])
        cer = np.mean([decoder.cer(true, pred) for true, pred in zip(target_strings, decoded_output)])
        val_stats['val_loss'].append(loss.item())
        val_stats['wer'].append(wer)
        val_stats['cer'].append(cer)
    for k, v in val_stats.items():
            val_stats[k] = np.mean(v)
    print(f'val_loss:{val_stats["val_loss"]}; val_wer:{val_stats["wer"]}; val_cer:{val_stats["cer"]}')
    file2.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='/home/silviu/Desktop/projects/ASR-main/configs/train_librispeech.yaml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(f))
    train(config)
