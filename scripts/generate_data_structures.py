import json
import sys
from pathlib import Path

import msgpack_numpy as mpn
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from toolz import curry
from tqdm import tqdm

sys.path.append('..')
from data.data import TxtLmdb, DetectFeatLmdb

mpn.patch()
import pandas as pd

import argparse


@curry
def get_bert_token_ids(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def add_bert_input_ids(test_df: pd.DataFrame):
    print(f"Generating BERT Token IDs for {len(test_df)} captions...")
    toker = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenizer = get_bert_token_ids(toker)
    test_df['input_ids'] = test_df['caption'].apply(tokenizer)
    return test_df


def add_img_fname(test_df: pd.DataFrame):
    print(f"Generating Feature File Names for {len(test_df)} WikiCaps IDs...")
    test_df['img_fname'] = test_df['wikicaps_id'].apply(get_feat_file_name)
    return test_df


def load_test_dataframe(wicsmmir_dir: str, version: str = "v1"):
    test_p = Path(wicsmmir_dir).joinpath(f"test_set_{version}.df.feather")
    print(f"Loading WikiCaps Test DataFrame from {test_p}")
    assert test_p.exists()
    return pd.read_feather(test_p)


def get_features_path(wicsmmir_dir: str):
    return Path(wicsmmir_dir).joinpath('features_36')


def get_feat_file_name(wid: int, wicsmmir_dir=None):
    fname = f"wikicaps_{wid}.npz"
    if wicsmmir_dir is not None:
        return get_features_path(wicsmmir_dir).joinpath(fname)
    return fname


def generate_text_image_json_mappings(test_df: pd.DataFrame, output_dir: str):
    # txt2img.json
    # structure: "<ID>": "<FEAT_FILE_NAME>.npz"
    txt2img_p = Path(output_dir).joinpath('txt_db/txt2img.json')
    if not txt2img_p.exists():
        txt2img_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating img2txts.json at {txt2img_p}")
    txt2img = {
        f"{wid}": f"{get_feat_file_name(wid)}" for wid in test_df['wikicaps_id']
    }

    with open(txt2img_p, "w", encoding="utf8") as fp:
        json.dump(txt2img, fp)

    # img2txts.json
    # structure: "<FEAT_FILE_NAME>.npz": ["ID"]
    img2txts_p = Path(output_dir).joinpath('txt_db/img2txts.json')
    if not img2txts_p.exists():
        img2txts_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating img2txts.json at {img2txts_p}")
    img2txts = {
        img: [txt] for txt, img in txt2img.items()
    }

    with open(img2txts_p, "w", encoding="utf8") as fp:
        json.dump(img2txts, fp)

    # id2len.json
    # structure: "<ID>": <len(BERT TOKEN IDS)>
    id2len_p = Path(output_dir).joinpath('txt_db/id2len.json')
    if not id2len_p.exists():
        id2len_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating id2len.json at {id2len_p}")
    id2len = {
        f"{row['wikicaps_id']}": len(row['input_ids']) for _, row in test_df.iterrows()
    }

    with open(id2len_p, "w", encoding="utf8") as fp:
        json.dump(id2len, fp)


def generate_text_lmdb(opts, test_df: pd.DataFrame):
    # we only need
    #   'img_fname' -> feature npz
    #   'input_ids' -> BERT Token IDs (without SEP and CLS etc)
    #   'raw' -> Raw caption (optional)
    out_p = Path(opts.output_dir).joinpath('txt_db/')
    if not out_p.exists():
        out_p.mkdir(parents=True, exist_ok=True)
    print(f"Generating TxtLmdb at {out_p}")
    test_df = add_bert_input_ids(test_df)
    test_df = add_img_fname(test_df)

    txt_lmdb = TxtLmdb(str(out_p), readonly=False)
    with tqdm(total=len(test_df)) as pbar:
        for _, row in test_df.iterrows():
            key = row['wikicaps_id']
            value = {'raw': row['caption'],
                     'input_ids': row['input_ids'],
                     'img_fname': row['img_fname']
                     }

            # store in TxtLmdb
            txt_lmdb[str(key)] = value
            pbar.update(1)


def generate_text_data(test_df: pd.DataFrame, opts):
    # generate the lmdb
    generate_text_lmdb(opts, test_df)

    # generate the json files
    generate_text_image_json_mappings(test_df, opts.output_dir)


def load_roi_feats(wicsmmir_dir: str, wid: int):
    fname_p = get_feat_file_name(wid, wicsmmir_dir)
    assert fname_p.exists()
    return np.load(fname_p, allow_pickle=True)


def get_norm_bb(bboxes, image_w, image_h):
    box_width = bboxes[:, 2] - bboxes[:, 0]
    box_height = bboxes[:, 3] - bboxes[:, 1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h
    scaled_x = bboxes[:, 0] / image_w
    scaled_y = bboxes[:, 1] / image_h

    box_width = box_width[..., np.newaxis]
    box_height = box_height[..., np.newaxis]
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]

    normalized_bbox = np.concatenate((scaled_x, scaled_y,
                                      scaled_x + scaled_width,
                                      scaled_y + scaled_height,
                                      scaled_width, scaled_height), axis=1)

    return normalized_bbox


def get_img_data_for_uniter(roi_feats):
    # we only need norm_bb and x (conf and soft_labels are not necessary because we have fixed num_bb = 36)
    bboxes = roi_feats['bbox']
    image_w = roi_feats['image_w']
    image_h = roi_feats['image_h']

    norm_bb = get_norm_bb(bboxes, image_w, image_h)
    features = roi_feats['x']

    uniter_data = {'norm_bb': norm_bb,
                   'features': features}

    return uniter_data


def generate_img_lmdb(opts, test_df):
    # we only need since we have fixed number of bboxes (=36)
    #   'features' -> features
    #   'norm_bb' -> normalized bbox

    out_p = Path(opts.output_dir).joinpath('img_db')
    if not out_p.exists():
        out_p.mkdir(parents=True, exist_ok=True)
    print(f"Generating DetectFeatLmdb at {out_p}")

    img_lmdb = DetectFeatLmdb(img_dir=str(out_p),
                              conf_th=-1,
                              max_bb=36,
                              min_bb=36,
                              num_bb=36,
                              compress=True,
                              readonly=False)
    with tqdm(total=len(test_df)) as pbar:
        for _, row in test_df.iterrows():
            key = get_feat_file_name(row['wikicaps_id'])
            # load the features from npz file
            roi_feats = load_roi_feats(opts.wicsmmir_dir, row['wikicaps_id'])
            # get uniter specific data structure
            value = get_img_data_for_uniter(roi_feats)

            # store in DetectFeatLmdb
            img_lmdb[str(key)] = value
            pbar.update(1)


def generate_img_data(test_df: pd.DataFrame, opts):
    # generate the lmdb
    generate_img_lmdb(opts, test_df)


def generate(opts):
    test_df = load_test_dataframe(opts.wicsmmir_dir)
    if opts.txt:
        print("Generating Text Data Structures")
        generate_text_data(test_df, opts)
    if opts.img:
        print("Generating Image Data Structures")
        generate_img_data(test_df, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--wicsmmir_dir", type=str, default="/srv/home/7schneid/data/TERAN_fork/data/wicsmmir",
                        help="The path to WICSMMIR features and DataFrames.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the generated data gets stored.")
    parser.add_argument("--txt", default=False, action='store_true',
                        help="If set, the Text Data Structures are generated.")
    parser.add_argument("--img", default=False, action='store_true',
                        help="If set, the Image Data Structures are generated.")

    opts = parser.parse_args()

    generate(opts)
