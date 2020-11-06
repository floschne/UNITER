#!/bin/bash

NGPU=1
UNITER_DATA_DIR=/home/p0w3r/gitrepos/UNITER_fork/downloads
FLICKR_30K_DATA_DIR=/home/p0w3r/datasets/flickr30k_images/flickr30k_images/


horovodrun -np $NGPU python image_retrieval.py  \
--query "$1" \
--img_feat_db ${UNITER_DATA_DIR}/img_db/flickr30k \
--img_ds ${FLICKR_30K_DATA_DIR} \
--checkpoint ${UNITER_DATA_DIR}/pretrained/uniter-base.pt \
--model_config ./config/uniter-base.json \
--meta_file ${UNITER_DATA_DIR}/txt_db/itm_flickr30k_test.db/meta.json \
--top_k 5 \
--bs 50 \
--num_imgs 1000 \
--fp16 \
--n_workers 0 \
--pin_mem
