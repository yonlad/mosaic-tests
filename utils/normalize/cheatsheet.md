 1. White bg (normalize.py) — participant images + resized_white_bg + flora-screenshots
  cd /Users/yonatan/Desktop/mosaic-tests/utils/normalize
  nohup python normalize.py \
    --bucket pistoletto.moe4 \
    --prefix selected-images/ \
    --dest-prefix normalized/ \
    --head-position 0.38 \
    > normalize_run.log 2>&1 &

  2. Black bg (normalize_black.py) — resized_black_bg
  nohup python normalize_black.py \
    --bucket pistoletto.moe4 \
    --dest-prefix normalized/ \
    --head-position 0.38 \
    > normalize_black_run.log 2>&1 &