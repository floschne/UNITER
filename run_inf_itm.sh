#!/bin/bash

NGPU=1
UNITER_DATA_DIR=/home/p0w3r/gitrepos/UNITER_fork/downloads/
ITM_RESULTS=${UNITER_DATA_DIR}/itm_results

horovodrun -np $NGPU python inf_itm.py  \
--txt_db ${UNITER_DATA_DIR}/txt_db/itm_flickr30k_test.db \
--img_db ${UNITER_DATA_DIR}/img_db/flickr30k \
--checkpoint ${UNITER_DATA_DIR}/pretrained/uniter-base.pt \
--model_config ./config/uniter-base.json \
--batch_size 50 \
--n_workers 1 \
--output_dir $ITM_RESULTS \
--fp16 \
--pin_mem
