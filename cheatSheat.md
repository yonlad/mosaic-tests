running full-body-mosaic-new.py:


python /Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/full-body-mosaic-new.py \
  --image "/Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/Jasmine copy.jpg" \
  --s3-prefix "selected-images/" \
  --base-cell-size 5 \
  --foreground-min-cell-size 25 \
  --background-min-cell-size 5 \
  --boundary-min-cell-size 20 \
  --detail-threshold 1.0 \
  --internal-thumbnail-size 256 256 \
  --exclude-white-background \
  --white-bg-luma-threshold 245 \
  --fg-mask-erosion-radius 2 \
  --no-blend \
  --thumbnail-limit 4000 \
  --max-thumbnail-usage 500 \
  --reuse-penalty 0.15 \
  --output-long-side 9000

  python /Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/full-body-mosaic-new.py \
  --image "/Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/Jasmine copy.jpg" \
  --s3-prefix "selected-images/" \
  --base-cell-size 4 \
  --foreground-min-cell-size 4 \
  --background-min-cell-size 4 \
  --boundary-min-cell-size 20 \
  --detail-threshold 1.0 \
  --internal-thumbnail-size 256 256 \
  --exclude-white-background \
  --white-bg-luma-threshold 245 \
  --fg-mask-erosion-radius 2 \
  --no-blend \
  --thumbnail-limit 4000 \
  --max-thumbnail-usage 500 \
  --reuse-penalty 0.15 \
  --output-long-side 9000