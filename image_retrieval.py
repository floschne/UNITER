import argparse
import pprint
import re
from pathlib import Path
from time import time
from typing import List

import numpy as np
import torch
from apex import amp
from horovod import torch as hvd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (PrefetchLoader,
                  DetectFeatLmdb,
                  ImageRetrievalDataset, itm_val_collate)
from model.itm import UniterForImageTextRetrieval
from utils.const import IMG_DIM
from utils.logger import LOGGER
from utils.misc import NoOp


def get_top_k_img_paths(scores, opts, ds: ImageRetrievalDataset) -> List[Path]:
    top_k_img_idx = torch.topk(scores, opts.top_k, dim=1).indices[0, :]
    if opts.top_k > 1:
        top_k_img_ids = np.array(ds.all_img_ids)[top_k_img_idx.cpu()]
    else:
        top_k_img_ids = [np.array(ds.all_img_ids)[top_k_img_idx.cpu()]]

    # get flickr30k ids
    prefix_pattern = re.compile(r"flickr30k_0*")
    suffix_pattern = re.compile(r"\.npz")
    ids = [str(suffix_pattern.sub("", prefix_pattern.sub("", top_k))) + ".jpg" for top_k in top_k_img_ids]

    paths = [Path(opts.img_ds).joinpath(i) for i in ids]
    assert all([p.exists() for p in paths]), "Cannot find images! Path to dataset correct? <" + opts.img_ds + ">"

    return paths


def run_retrieval(opts):
    # horovod init
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
        device, n_gpu, hvd.rank(), opts.fp16))

    # load images lmdb
    img_feat_db = DetectFeatLmdb(opts.img_feat_db, compress=False)

    # create retrieval dataset for input
    img_ret_ds = ImageRetrievalDataset(opts.query,
                                       img_feat_db,
                                       opts.meta_file,
                                       opts.bs,
                                       opts.num_imgs)
    # init dataloader
    dataloader = DataLoader(img_ret_ds,
                            batch_size=1,  # TODO why always 1 ? taken from inf_itm.py
                            # if this is non-zero a deadlock occurs https://github.com/pytorch/pytorch/issues/1355
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem,
                            collate_fn=itm_val_collate)
    dataloader = PrefetchLoader(dataloader)

    # init model
    model = UniterForImageTextRetrieval.from_pretrained(opts.model_config,
                                                        torch.load(opts.checkpoint),
                                                        img_dim=IMG_DIM)
    model.init_output()  # zero shot setting
    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    scores = run_img_retrieval(model, dataloader)
    if hvd.rank() == 0:
        top_k_paths = get_top_k_img_paths(scores, opts, dataloader.dataset)

        results = pprint.pformat(top_k_paths, indent=4)
        LOGGER.info(
            "\n======================== Results =========================\n"
            f"{results}"
            "\n==========================================================\n"
        )

        return top_k_paths
    else:
        return []


@torch.no_grad()
def run_img_retrieval(model, dataloader):
    model.eval()
    st = time()
    LOGGER.info(f"start running Image Retrieval on {len(dataloader.dataset.all_img_ids)} images...")

    score_matrix = torch.zeros(len(dataloader.dataset),
                               len(dataloader.dataset.all_img_ids),
                               device=torch.device("cuda"),
                               dtype=torch.float16)

    for i, mini_batches in enumerate(dataloader):
        j = 0

        if hvd.rank() == 0:
            pbar = tqdm(total=len(dataloader.dataset.all_img_ids), desc="Mini-Batches")
        else:
            pbar = NoOp()

        for batch in mini_batches:
            scores = model(batch, compute_loss=False)
            bs = scores.size(0)
            score_matrix.data[i, j:j + bs] = scores.data.squeeze(1).half()
            j += bs
            pbar.update(len(batch['input_ids']))
        assert j == score_matrix.size(1)
    model.train()

    # all_score is the similarity matrix from input to images in the ds
    all_score = hvd.allgather(score_matrix)
    assert all_score.size() == (hvd.size(), len(dataloader.dataset.all_img_ids)), \
        f"{all_score.size()} == {(hvd.size(), len(dataloader.dataset.all_img_ids))}"

    # TODO do we need this? learn horovod / mpi
    if hvd.rank() != 0:
        return {}, tuple()

    tot_time = time() - st
    print(f"Image Retrieval finished in {int(tot_time)} seconds, ")

    return all_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--query", default=None, type=str,
                        help="Textual Query for Image Retrieval")
    parser.add_argument("--img_feat_db", default=None, type=str,
                        help="path to image feature lmdb")
    parser.add_argument("--img_ds", default=None, type=str,
                        help="path to 'raw' Image dataset")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint binary")
    parser.add_argument("--model_config", default=None, type=str,
                        help="path to model config json")
    parser.add_argument("--meta_file", default=None, type=str,
                        help="path to meta file in json format")

    # optional parameters
    parser.add_argument('--top_k', type=int, default=5,
                        help='Returns the top_k images matching the query.')
    # set to higher value like 400 if you use 'proper' GPUs
    parser.add_argument('--bs', type=int, default=50,
                        help='batch size')
    parser.add_argument('--num_imgs', type=int, default=0,
                        help='Number of images used to compute scores. If 0 ALL Flickr30k images are used!')

    # (optional) device parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=0,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    run_retrieval(args)
