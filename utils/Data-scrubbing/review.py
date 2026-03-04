#!/usr/bin/env python3
"""
S3 Image Review Tool

Lists every image under selected-images/ in the target S3 bucket,
cross-references with DynamoDB blend records, and generates an
interactive HTML gallery for manual quality review.

The HTML page lets you visually flag low-quality images, then
export a JSON deletion manifest consumed by delete.py.

Usage:
    python review.py                 # defaults to blend 1
    python review.py --blend 2
    python review.py --blend 1 --prefix "selected-images/some-subfolder/"
"""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from config import (
    get_s3_client,
    get_dynamodb_resource,
    get_blend_config,
    IMAGE_PREFIX,
    IMAGE_EXTENSIONS,
)

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def list_s3_images(s3_client, bucket: str, prefix: str) -> list[dict]:
    """Return metadata for every image object under *prefix*."""
    images = []
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}

    print(f"Listing objects in s3://{bucket}/{prefix} ...")

    while True:
        response = s3_client.list_objects_v2(**kwargs)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
            if ext not in IMAGE_EXTENSIONS:
                continue
            images.append({
                "key": key,
                "size": obj["Size"],
                "size_kb": round(obj["Size"] / 1024, 1),
                "last_modified": obj["LastModified"].isoformat(),
            })

        if response.get("IsTruncated"):
            kwargs["ContinuationToken"] = response["NextContinuationToken"]
        else:
            break

    print(f"  Found {len(images)} images")
    return images


def add_presigned_urls(s3_client, bucket: str, images: list[dict],
                       expiry: int = 86400) -> None:
    """Mutate each dict in *images* to include a presigned URL (default 24 h)."""
    print(f"Generating presigned URLs (expiry {expiry}s) ...")
    for img in images:
        img["url"] = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": img["key"]},
            ExpiresIn=expiry,
        )


# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def scan_dynamodb(dynamodb, table_name: str) -> list[dict]:
    """Full scan of the blend table. Returns raw items."""
    print(f"Scanning DynamoDB table: {table_name} ...")
    table = dynamodb.Table(table_name)
    items: list[dict] = []
    scan_kwargs: dict = {}

    while True:
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    print(f"  Found {len(items)} blend records")
    return items


def _normalize_key(raw_key: str) -> str:
    """Extract the selected-images/… portion no matter how the key is stored."""
    idx = raw_key.find("selected-images/")
    if idx != -1:
        return raw_key[idx:]
    return raw_key


def build_image_blend_map(images: list[dict], blends: list[dict]) -> dict:
    """Map S3 key → list of blend_id strings that reference it."""
    norm_to_s3: dict[str, str] = {}
    for img in images:
        norm_to_s3[_normalize_key(img["key"])] = img["key"]

    mapping: dict[str, list[str]] = {}

    for blend in blends:
        blend_id = blend.get("blend_id", "unknown")
        for field in ("source_image_key", "random_image_key"):
            db_val = blend.get(field, "")
            if not db_val:
                continue
            norm = _normalize_key(db_val)
            s3_key = norm_to_s3.get(norm)
            if s3_key:
                mapping.setdefault(s3_key, [])
                if blend_id not in mapping[s3_key]:
                    mapping[s3_key].append(blend_id)

    return mapping


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_images_by_session(images: list[dict]) -> dict[str, list[dict]]:
    """Group images by their parent folder (session / user-id)."""
    groups: dict[str, list[dict]] = {}
    for img in images:
        parts = img["key"].split("/")
        session = parts[-2] if len(parts) >= 3 else "ungrouped"
        groups.setdefault(session, []).append(img)
    return groups


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Image Review – __BUCKET__</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     background:#0f0f1a;color:#e0e0e0;min-height:100vh}
header{position:sticky;top:0;z-index:100;background:#14142a;
       border-bottom:1px solid #2a2a4a;padding:12px 24px;display:flex;
       flex-wrap:wrap;align-items:center;gap:12px}
header h1{font-size:1.1rem;font-weight:600;white-space:nowrap}
header .stats{font-size:.85rem;color:#888;white-space:nowrap}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-left:auto}
.controls button{padding:6px 14px;border:1px solid #3a3a5a;border-radius:6px;
                  background:#1e1e38;color:#ccc;cursor:pointer;font-size:.82rem;
                  transition:all .15s}
.controls button:hover{background:#2a2a50;border-color:#6c63ff;color:#fff}
.controls button.danger{border-color:#ff4444;color:#ff6b6b}
.controls button.danger:hover{background:#ff4444;color:#fff}
.controls .counter{font-size:.85rem;color:#ff6b6b;font-weight:600;min-width:100px}
.filter-bar{padding:8px 24px;background:#12122a;border-bottom:1px solid #1e1e38;
            display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.filter-bar input{padding:6px 12px;border:1px solid #2a2a4a;border-radius:6px;
                   background:#1a1a2e;color:#e0e0e0;font-size:.85rem;width:260px}
.filter-bar select{padding:6px 10px;border:1px solid #2a2a4a;border-radius:6px;
                    background:#1a1a2e;color:#e0e0e0;font-size:.85rem}
.filter-bar label{font-size:.82rem;color:#888}

main{padding:16px 24px 60px}
.session-group{margin-bottom:28px}
.session-header{display:flex;align-items:center;gap:10px;padding:8px 0;
                cursor:pointer;user-select:none}
.session-header h2{font-size:.92rem;font-weight:500;color:#aaa}
.session-header .badge{font-size:.75rem;background:#2a2a4a;padding:2px 8px;
                       border-radius:10px;color:#888}
.session-header .toggle{color:#555;transition:transform .2s}
.session-header.collapsed .toggle{transform:rotate(-90deg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
      gap:12px;padding:8px 0}

.card{position:relative;border-radius:8px;overflow:hidden;
      border:2px solid transparent;cursor:pointer;
      transition:border-color .15s,box-shadow .15s;background:#1a1a2e}
.card:hover{border-color:#3a3a5a;box-shadow:0 2px 12px rgba(0,0,0,.4)}
.card.flagged{border-color:#ff4444}
.card.flagged::after{content:"✕";position:absolute;top:8px;right:8px;
                     width:28px;height:28px;background:#ff4444;color:#fff;
                     border-radius:50%;display:flex;align-items:center;
                     justify-content:center;font-size:14px;font-weight:700}
.card img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;
          background:#111}
.card .meta{padding:6px 10px;font-size:.75rem;color:#888;
            display:flex;justify-content:space-between;align-items:center}
.card .meta .blends{background:#6c63ff22;color:#6c63ff;padding:1px 6px;
                    border-radius:4px;font-size:.7rem}

.lightbox{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.92);
          align-items:center;justify-content:center;cursor:zoom-out}
.lightbox.open{display:flex}
.lightbox img{max-width:92vw;max-height:92vh;border-radius:6px}
.lightbox .info{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
                background:#1a1a2e;padding:8px 18px;border-radius:8px;
                font-size:.82rem;color:#aaa;white-space:nowrap}

.toast{position:fixed;bottom:24px;right:24px;background:#1e1e38;
       border:1px solid #6c63ff;border-radius:8px;padding:12px 20px;
       font-size:.85rem;color:#ccc;z-index:300;opacity:0;
       transition:opacity .3s;pointer-events:none}
.toast.show{opacity:1}
</style>
</head>
<body>

<header>
  <h1>Image Review</h1>
  <span class="stats" id="stats"></span>
  <div class="controls">
    <button onclick="selectAll()">Select All</button>
    <button onclick="deselectAll()">Deselect All</button>
    <button onclick="invertSelection()">Invert</button>
    <button onclick="selectSmall()">Flag &lt; 50 KB</button>
    <span class="counter" id="counter">0 flagged</span>
    <button class="danger" onclick="exportManifest()">Export Deletion Manifest</button>
  </div>
</header>

<div class="filter-bar">
  <input id="search" type="text" placeholder="Filter by session ID or filename…"
         oninput="applyFilter()" />
  <label>Sort:
    <select id="sortBy" onchange="renderGallery()">
      <option value="session">Session</option>
      <option value="size_asc">Size ↑</option>
      <option value="size_desc">Size ↓</option>
      <option value="date_asc">Date ↑</option>
      <option value="date_desc">Date ↓</option>
      <option value="name">Filename</option>
    </select>
  </label>
  <label>
    <input type="checkbox" id="showFlaggedOnly" onchange="applyFilter()" />
    Show flagged only
  </label>
</div>

<main id="gallery"></main>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lbImg" src="" />
  <div class="info" id="lbInfo"></div>
</div>
<div class="toast" id="toast"></div>

<script>
const RAW_DATA  = __IMAGE_DATA__;
const BUCKET    = "__BUCKET__";
const TABLE     = "__TABLE__";
const BLEND_NUM = __BLEND_NUM__;

const flagged = new Set();

function filename(key){ return key.split("/").pop(); }
function sessionId(key){
  const parts = key.split("/");
  return parts.length >= 3 ? parts[parts.length - 2] : "ungrouped";
}

function updateCounter(){
  document.getElementById("counter").textContent = flagged.size + " flagged";
}
function toast(msg){
  const el = document.getElementById("toast");
  el.textContent = msg; el.classList.add("show");
  setTimeout(()=> el.classList.remove("show"), 2500);
}

function toggle(key){
  flagged.has(key) ? flagged.delete(key) : flagged.add(key);
  const card = document.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if(card) card.classList.toggle("flagged", flagged.has(key));
  updateCounter();
}

function selectAll(){ getVisible().forEach(i=>{ flagged.add(i.key); syncCard(i.key); }); updateCounter(); }
function deselectAll(){ flagged.clear(); document.querySelectorAll(".card.flagged").forEach(c=>c.classList.remove("flagged")); updateCounter(); }
function invertSelection(){ getVisible().forEach(i=>{ flagged.has(i.key)?flagged.delete(i.key):flagged.add(i.key); syncCard(i.key); }); updateCounter(); }
function selectSmall(){ getVisible().filter(i=>i.size_kb<50).forEach(i=>{ flagged.add(i.key); syncCard(i.key); }); updateCounter(); toast("Flagged images under 50 KB"); }
function syncCard(key){ const c=document.querySelector(`[data-key="${CSS.escape(key)}"]`); if(c) c.classList.toggle("flagged",flagged.has(key)); }

function getVisible(){
  const q = document.getElementById("search").value.toLowerCase();
  const fo = document.getElementById("showFlaggedOnly").checked;
  return RAW_DATA.filter(i=>{
    if(fo && !flagged.has(i.key)) return false;
    if(q && !i.key.toLowerCase().includes(q)) return false;
    return true;
  });
}

function applyFilter(){ renderGallery(); }

function sorted(items){
  const s = document.getElementById("sortBy").value;
  const copy = [...items];
  if(s==="size_asc")  copy.sort((a,b)=>a.size-b.size);
  if(s==="size_desc") copy.sort((a,b)=>b.size-a.size);
  if(s==="date_asc")  copy.sort((a,b)=>a.last_modified.localeCompare(b.last_modified));
  if(s==="date_desc") copy.sort((a,b)=>b.last_modified.localeCompare(a.last_modified));
  if(s==="name")      copy.sort((a,b)=>filename(a.key).localeCompare(filename(b.key)));
  return copy;
}

function renderGallery(){
  const items = sorted(getVisible());
  const gallery = document.getElementById("gallery");
  const groups = {};
  items.forEach(i=>{
    const sid = sessionId(i.key);
    (groups[sid] = groups[sid]||[]).push(i);
  });

  let html = "";
  for(const [sid, imgs] of Object.entries(groups)){
    html += `<div class="session-group">
      <div class="session-header" onclick="this.classList.toggle('collapsed');this.nextElementSibling.style.display=this.classList.contains('collapsed')?'none':'grid'">
        <span class="toggle">▼</span>
        <h2>${sid}</h2>
        <span class="badge">${imgs.length} image${imgs.length!==1?"s":""}</span>
      </div>
      <div class="grid">`;
    for(const img of imgs){
      const f = flagged.has(img.key) ? " flagged" : "";
      const blendCount = img.blend_ids ? img.blend_ids.length : 0;
      const blendBadge = blendCount ? `<span class="blends">${blendCount} blend${blendCount!==1?"s":""}</span>` : "";
      html += `<div class="card${f}" data-key="${img.key}" onclick="toggle('${img.key.replace(/'/g,"\\'")}')" oncontextmenu="event.preventDefault();openLightbox('${img.url.replace(/'/g,"\\'")}','${img.key.replace(/'/g,"\\'")}')">
        <img src="${img.url}" loading="lazy" alt="${filename(img.key)}" />
        <div class="meta"><span>${filename(img.key)} · ${img.size_kb} KB</span>${blendBadge}</div>
      </div>`;
    }
    html += `</div></div>`;
  }
  gallery.innerHTML = html;

  const totalSessions = Object.keys(groups).length;
  document.getElementById("stats").textContent =
    `${BUCKET} · Blend ${BLEND_NUM} · ${items.length} images · ${totalSessions} sessions`;
}

function openLightbox(url, key){
  document.getElementById("lbImg").src = url;
  document.getElementById("lbInfo").textContent = key;
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox(){
  document.getElementById("lightbox").classList.remove("open");
  document.getElementById("lbImg").src = "";
}
document.addEventListener("keydown", e=>{ if(e.key==="Escape") closeLightbox(); });

function exportManifest(){
  if(flagged.size===0){ toast("Nothing flagged – select images first"); return; }
  const manifest = {
    created_at: new Date().toISOString(),
    blend_number: BLEND_NUM,
    bucket: BUCKET,
    table: TABLE,
    items: [...flagged].map(key=>{
      const img = RAW_DATA.find(i=>i.key===key);
      return {
        s3_key: key,
        blend_ids: img?.blend_ids || [],
      };
    }),
  };
  const blob = new Blob([JSON.stringify(manifest, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `deletion_manifest_blend${BLEND_NUM}_${Date.now()}.json`;
  a.click();
  toast(`Exported ${flagged.size} item(s)`);
}

renderGallery();
</script>
</body>
</html>"""


def generate_html(images: list[dict], bucket: str, table: str,
                  blend_num: int) -> str:
    data_json = json.dumps(images, default=str)
    html = HTML_TEMPLATE
    html = html.replace("__IMAGE_DATA__", data_json)
    html = html.replace("__BUCKET__", bucket)
    html = html.replace("__TABLE__", table)
    html = html.replace("__BLEND_NUM__", str(blend_num))
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate an image-review gallery")
    p.add_argument("--blend", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="Blend number (1-4, default 1)")
    p.add_argument("--prefix", default=None,
                   help=f"S3 prefix override (default: {IMAGE_PREFIX})")
    p.add_argument("--output", default=None,
                   help="Output HTML path (default: review_blend<N>.html in this dir)")
    p.add_argument("--no-open", action="store_true",
                   help="Don't auto-open the HTML in a browser")
    p.add_argument("--url-expiry", type=int, default=86400,
                   help="Presigned URL expiry in seconds (default: 86400 = 24 h)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_blend_config(args.blend)
    bucket = cfg["bucket"]
    table = cfg["table"]
    prefix = args.prefix or IMAGE_PREFIX

    print(f"{'='*60}")
    print(f"  Blend {args.blend}:  bucket={bucket}  table={table}")
    print(f"  Prefix: {prefix}")
    print(f"{'='*60}\n")

    s3 = get_s3_client()
    dynamodb = get_dynamodb_resource()

    # 1) List images
    images = list_s3_images(s3, bucket, prefix)
    if not images:
        print("No images found. Nothing to review.")
        sys.exit(0)

    # 2) Presigned URLs
    add_presigned_urls(s3, bucket, images, expiry=args.url_expiry)

    # 3) Cross-reference with DynamoDB
    blends = scan_dynamodb(dynamodb, table)
    blend_map = build_image_blend_map(images, blends)

    orphan_count = 0
    for img in images:
        img["blend_ids"] = blend_map.get(img["key"], [])
        if not img["blend_ids"]:
            orphan_count += 1

    print(f"\n  {len(images)} images total")
    print(f"  {len(images) - orphan_count} linked to blend records")
    print(f"  {orphan_count} orphaned (no blend reference)\n")

    # 4) Generate HTML
    html = generate_html(images, bucket, table, args.blend)

    out_dir = Path(__file__).parent
    out_path = Path(args.output) if args.output else out_dir / f"review_blend{args.blend}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Gallery saved to: {out_path.resolve()}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
        print("Opened in browser.")

    print("\nWorkflow:")
    print("  1. Click images to flag low-quality assets (right-click for full-size)")
    print("  2. Use toolbar buttons to bulk-select (e.g. 'Flag < 50 KB')")
    print("  3. Click 'Export Deletion Manifest' to download the JSON")
    print("  4. Run:  python delete.py --manifest <downloaded_file>.json --dry-run")
    print("  5. When satisfied:  python delete.py --manifest <downloaded_file>.json")


if __name__ == "__main__":
    main()
