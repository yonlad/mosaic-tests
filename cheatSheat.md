running full-body-mosaic-new.py:

##high res version:
python /Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/full-body-mosaic-new.py \
  --image "/Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/Jasmine copy.jpg" \
  --s3-prefix "selected-images/" \
  --base-cell-size 5 \
  --foreground-min-cell-size 25 \
  --background-min-cell-size 5 \
  --boundary-min-cell-size 20 \
  --detail-threshold 1.0 \
  --internal-thumbnail-size 480 480 \
  --source-downscale-long-side -1 \
  --use-full-res-thumbs \
  --exclude-white-background \
  --output-format PNG \
  --white-bg-luma-threshold 245 \
  --fg-mask-erosion-radius 2 \
  --no-blend \
  --thumbnail-limit 4000 \
  --max-thumbnail-usage 500 \
  --reuse-penalty 0.15 \
  --output-long-side 9000 



##>300dpi version:
python /Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/full-body-mosaic-new.py \
  --image "/Users/ledi/Desktop/WORK/DMINTI/mosaic-tests/Jasmine copy.jpg" \
  --s3-prefix "selected-images/" \
  --base-cell-size 5 \
  --foreground-min-cell-size 25 \
  --background-min-cell-size 5 \
  --boundary-min-cell-size 20 \
  --detail-threshold 1.0 \
  --internal-thumbnail-size 1000 1000 \
  --exclude-white-background \
  --white-bg-luma-threshold 245 \
  --fg-mask-erosion-radius 2 \
  --no-blend \
  --output-format PNG \
  --thumbnail-limit 4000 \
  --max-thumbnail-usage 100 \
  --reuse-penalty 0.15 \
  --output-long-side 40000


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