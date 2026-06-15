# Task: Transparent Thumbnail Backgrounds in Mosaic Generation

## Goal

In `full-body-mosaic-transparent.py`, when building a mosaic from thumbnail photos, we want each thumbnail's **white background to become transparent** so the **canvas background color** (grey, mirror texture, etc.) bleeds through the gaps between tiles. However, the **person content** inside each thumbnail — including white/light clothing, pale skin, etc. — must be **fully preserved**.

## The Problem

The thumbnail images in S3 (`selected-images/` prefix) are photos of people on **pure white backgrounds** (255,255,255). When these thumbnails are placed as mosaic tiles, their white backgrounds create a solid white fill that hides the canvas color.

We want:
- Canvas background visible through the thumbnail's background areas (the bleed-through effect)
- Person content (face, hair, clothing — even white shirts) fully preserved and visible
- Clean separation between "thumbnail background" and "thumbnail subject content"

## What We Tried (and why each failed)

### 1. Global white threshold (`--transparent-thumbnail-bg --thumbnail-white-threshold 240`)
Made ALL pixels with R,G,B > 240 transparent in every thumbnail.
- **Result**: Bleed-through effect worked great for dark areas (hair, dark clothing). But white shirts, light skin, and any bright content ALSO became transparent and disappeared into the canvas.
- **Why**: Can't distinguish "white background" from "white clothing" by color alone.

### 2. Flood-fill from thumbnail edges
Used `scipy.ndimage.label` to find connected white regions touching the image border, made only those transparent.
- **Result**: Still failed for white shirts. The white clothing in the thumbnail connects to the white border through collar/edges, so flood-fill leaks through the shirt and removes it.
- **Why**: White clothing often physically connects to the white background at the garment edges.

### 3. Higher threshold (252-254)
Raised the global threshold to only catch near-pure-white pixels.
- **Result**: No visible improvement. The source photos have backgrounds that are exactly 255,255,255, but the thumbnails matched for light areas are also almost entirely pure white.
- **Why**: The matching algorithm selects light-colored thumbnails for light source areas; those thumbnails are mostly white background with a small person figure.

### 4. Adaptive per-cell transparency
Applied transparency only to dark cells (luma < threshold) and pasted solid thumbnails for light cells (shirts).
- **Result**: Shirts partially preserved, but created visible white contour artifacts at the boundary between transparent and solid regions. Looked unnatural.
- **Why**: Sharp binary switch between transparent and solid tiles creates a visible edge.

## What DID Work Well

- **`--fg-mask-method rembg`**: Using rembg for the **source image** foreground mask works perfectly. It correctly identifies both people (including white shirts) as foreground, keeping the background tile-free. This should be kept.
- **`--no-exclude-white-background`**: Combined with rembg mask, this ensures tiles ARE placed in white shirt areas.
- **`--fg-mask-erosion-radius 0 --boundary-min-cell-size 5`**: These settings preserve hair detail at the edges.

## Recommended Approach: Pre-process Thumbnails with rembg

The only robust way to distinguish "white background" from "white content" in a thumbnail is **semantic segmentation**. Since all thumbnails are clean photos of people on white backgrounds, rembg handles them extremely well.

### Implementation Plan

1. **Create a thumbnail pre-processing cache**:
   - On first run with `--transparent-thumbnail-bg`, run rembg on each downloaded thumbnail
   - Save the resulting RGBA image (with proper alpha) to a local cache directory (e.g., `utils/mockup-pipeline/thumbnail_cache/` or a configurable path)
   - Cache key: S3 object key hash or the key path itself
   - On subsequent runs, load from cache instead of re-running rembg

2. **Modify `_download_single_image_s3`**:
   - After downloading, check cache for pre-processed RGBA version
   - If not cached: run `rembg.remove(img)` to get RGBA with proper person mask
   - Save to cache
   - Use the RGBA thumbnail for placement

3. **Modify `match_and_place`**:
   - When `transparent_thumbnail_bg` is enabled, thumbnails are already RGBA with proper alpha
   - Paste with alpha mask: `canvas.paste(tile_img, (ox, oy), tile_img)`
   - The person content (including white shirts) stays opaque
   - Only the actual background is transparent, showing the canvas through

4. **Performance considerations**:
   - rembg takes ~1-2 seconds per image on CPU
   - For 4000 thumbnails: ~60-120 minutes first run, instant on subsequent runs (cached)
   - Consider: batch processing with progress bar, optional `--rebuild-cache` flag
   - The cache makes this a one-time cost

### Alternative: Lighter segmentation
If rembg is too slow, consider:
- `rembg` with the `u2netp` model (smaller, faster, slightly less accurate)
- Pre-computing all thumbnail masks as a separate batch script that runs overnight
- Using `--thumbnail-limit` to reduce the number during testing

## Key Files

| File | Role |
|------|------|
| `full-body-mosaic-transparent.py` | Main mosaic script to modify |
| `full-body-mosaic-new.py` | Reference — original non-transparent version |
| `mosaics-grey.py` | Reference — simpler mosaic with grey background |
| `utils/backgrounds/mirror-background.png` | Mirror texture for canvas background |
| `.env` | AWS credentials for S3 thumbnail access |

## Current Config That Works (except thumbnail transparency)

```bash
python full-body-mosaic-transparent.py \
    --image "full-bod-imgs/prototypes/new.png" \
    --s3-prefix "selected-images/" \
    --base-cell-size 5 \
    --foreground-min-cell-size 25 \
    --background-min-cell-size 5 \
    --boundary-min-cell-size 5 \
    --detail-threshold 1.0 \
    --internal-thumbnail-size 1000 1000 \
    --fg-mask-method rembg \
    --no-exclude-white-background \
    --fg-mask-erosion-radius 0 \
    --no-blend \
    --output-format PNG \
    --thumbnail-limit 4000 \
    --max-thumbnail-usage 100 \
    --reuse-penalty 0.15 \
    --output-long-side 40000 \
    --create-grey-background \
    --transparent-thumbnail-bg
```

## Test Image

Use `full-bod-imgs/prototypes/new.png` — two women facing each other, both wearing white shirts on a white background. This is the hardest case because the shirts are white-on-white.

## Success Criteria

1. Dark areas (hair, face, dark clothing) show the canvas background through the thumbnail's white areas — the "bleed-through" mosaic effect
2. White/light areas (shirts, pale skin) are fully visible with tile content preserved — NO disappearing shirts
3. Background between the two people is clean canvas color — no stray tiles
4. No visible contour artifacts at the boundary between dark and light regions
5. Hair of both figures (including the left woman's thin ponytail/bangs) is fully filled with tiles
