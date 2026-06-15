#!/usr/bin/env python3
"""
DynamoDB Blend Review Tool

Scans every blend record in the target DynamoDB table and generates an
interactive HTML gallery showing the three assets side by side:
  1. Source image
  2. Random image
  3. Video blend (HTML5 <video>)

The HTML page lets you visually flag low-quality blends, then export a
JSON deletion manifest.

Usage (CLI):
    python review_blends.py --blend 1
    python review_blends.py --blend 2

Usage (from launcher):
    from review_blends import run_review_blends
    path = run_review_blends(blend_number=1, on_progress=print)
"""

import argparse
import json
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import (
    get_s3_client,
    get_dynamodb_resource,
    get_blend_config,
    app_data_dir,
    BLENDS,
)


def _emit(msg, callback=None):
    print(msg)
    if callback:
        callback(msg)


# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def scan_dynamodb(dynamodb, table_name: str, on_progress=None) -> list[dict]:
    _emit(f"Scanning DynamoDB table: {table_name} ...", on_progress)
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

    _emit(f"  Found {len(items)} blend records", on_progress)
    return items


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

def _normalize_key(raw_key: str, bucket: str) -> str:
    parts = raw_key.split("/")
    for i, part in enumerate(parts):
        if part.startswith("selected-images") or part.startswith("video-blends"):
            return "/".join(parts[i:])
    prefix = bucket + "/"
    if raw_key.startswith(prefix):
        return raw_key[len(prefix):]
    return raw_key


# ---------------------------------------------------------------------------
# Presigned URL generation
# ---------------------------------------------------------------------------

def _find_asset_bucket(s3_client, key: str, primary_bucket: str) -> str:
    if not key:
        return ""
    try:
        s3_client.head_object(Bucket=primary_bucket, Key=key)
        return primary_bucket
    except Exception:
        pass
    for cfg in BLENDS.values():
        b = cfg["bucket"]
        if b == primary_bucket:
            continue
        try:
            s3_client.head_object(Bucket=b, Key=key)
            return b
        except Exception:
            continue
    return ""


def _process_one_item(s3_client, bucket, item, expiry):
    blend_id = item.get("blend_id", "")
    if not blend_id:
        return None

    source_key_raw = item.get("source_image_key", "")
    random_key_raw = item.get("random_image_key", "")
    video_key_raw = item.get("s3_video_key", "") or item.get("s3_key", "")

    source_key = _normalize_key(source_key_raw, bucket) if source_key_raw else ""
    random_key = _normalize_key(random_key_raw, bucket) if random_key_raw else ""
    video_key = _normalize_key(video_key_raw, bucket) if video_key_raw else ""

    source_bucket = _find_asset_bucket(s3_client, source_key, bucket) if source_key else ""
    random_bucket = _find_asset_bucket(s3_client, random_key, bucket) if random_key else ""
    video_bucket = _find_asset_bucket(s3_client, video_key, bucket) if video_key else ""

    def presign(key, target_bucket):
        if not key or not target_bucket:
            return ""
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": target_bucket, "Key": key},
            ExpiresIn=expiry,
        )

    cross = 1 if (random_bucket and random_bucket != bucket) else 0

    return {
        "blend_id": blend_id,
        "source_key": source_key,
        "random_key": random_key,
        "video_key": video_key,
        "source_url": presign(source_key, source_bucket),
        "random_url": presign(random_key, random_bucket),
        "video_url": presign(video_key, video_bucket),
        "status": item.get("status", ""),
        "created_at": item.get("created_at", ""),
        "user_id": item.get("user_id", ""),
        "email": item.get("email", ""),
        "_cross": cross,
    }


def build_blend_records(s3_client, bucket: str, items: list[dict],
                        expiry: int = 86400, on_progress=None) -> list[dict]:
    _emit(f"Building blend records with presigned URLs (expiry {expiry}s) ...", on_progress)
    _emit(f"  (probing across {len(BLENDS)} buckets — using 20 parallel threads)", on_progress)

    records = []
    cross_bucket_count = 0
    done_count = 0
    total = len(items)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {
            pool.submit(_process_one_item, s3_client, bucket, item, expiry): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(futures):
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                _emit(f"  Processed {done_count}/{total} records ...", on_progress)

            result = future.result()
            if result is None:
                continue
            cross_bucket_count += result.pop("_cross")
            records.append(result)

    _emit(f"  Built {len(records)} review records", on_progress)
    if cross_bucket_count:
        _emit(f"  {cross_bucket_count} random images resolved from other buckets", on_progress)
    return records


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Blend Review - __BUCKET__</title>
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
.filter-bar input[type="text"]{padding:6px 12px;border:1px solid #2a2a4a;border-radius:6px;
                   background:#1a1a2e;color:#e0e0e0;font-size:.85rem;width:300px}
.filter-bar select{padding:6px 10px;border:1px solid #2a2a4a;border-radius:6px;
                    background:#1a1a2e;color:#e0e0e0;font-size:.85rem}
.filter-bar label{font-size:.82rem;color:#888;display:flex;align-items:center;gap:4px}

main{padding:16px 24px 60px}

.blend-card{position:relative;background:#1a1a2e;border:2px solid transparent;
            border-radius:10px;margin-bottom:16px;overflow:hidden;
            cursor:pointer;transition:border-color .15s,box-shadow .15s}
.blend-card:hover{border-color:#3a3a5a;box-shadow:0 2px 12px rgba(0,0,0,.4)}
.blend-card.flagged{border-color:#ff4444}
.blend-card.flagged::after{content:"FLAGGED";position:absolute;top:12px;right:12px;
                           background:#ff4444;color:#fff;padding:4px 12px;
                           border-radius:4px;font-size:.75rem;font-weight:700;
                           letter-spacing:.5px;z-index:10}

.blend-assets{display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;
              background:#0f0f1a}
.asset-cell{position:relative;background:#111;min-height:200px;display:flex;
            flex-direction:column}
.asset-cell .label{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.7);
                   color:#aaa;padding:2px 8px;border-radius:4px;font-size:.7rem;
                   font-weight:600;letter-spacing:.5px;text-transform:uppercase;z-index:5}
.asset-cell img{width:100%;height:250px;object-fit:cover;display:block}
.asset-cell video{width:100%;height:250px;object-fit:cover;display:block;background:#000}
.asset-cell .no-asset{width:100%;height:250px;display:flex;align-items:center;
                      justify-content:center;color:#555;font-size:.8rem}

.blend-meta{padding:10px 16px;display:flex;flex-wrap:wrap;gap:12px;
            font-size:.78rem;color:#888;border-top:1px solid #2a2a4a}
.blend-meta .tag{background:#2a2a4a;padding:2px 8px;border-radius:4px}
.blend-meta .status-completed{color:#4ade80}
.blend-meta .status-failed{color:#ff6b6b}
.blend-meta .status-pending{color:#fbbf24}

.lightbox{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.92);
          align-items:center;justify-content:center;cursor:zoom-out}
.lightbox.open{display:flex}
.lightbox img,
.lightbox video{max-width:92vw;max-height:92vh;border-radius:6px}
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
  <h1>Blend Review</h1>
  <span class="stats" id="stats"></span>
  <div class="controls">
    <button id="btnSelectAll">Select All</button>
    <button id="btnDeselect">Deselect All</button>
    <button id="btnInvert">Invert</button>
    <span class="counter" id="counter">0 flagged</span>
    <button class="danger" id="btnExport">Export Deletion Manifest</button>
  </div>
</header>

<div class="filter-bar">
  <input id="search" type="text" placeholder="Filter by blend ID, user ID, or email..." />
  <label>Sort:
    <select id="sortBy">
      <option value="date_desc">Date (newest)</option>
      <option value="date_asc">Date (oldest)</option>
      <option value="status">Status</option>
      <option value="user_id">User ID</option>
      <option value="blend_id">Blend ID</option>
    </select>
  </label>
  <label>
    <input type="checkbox" id="showFlaggedOnly" />
    Show flagged only
  </label>
</div>

<main id="gallery"></main>

<div class="lightbox" id="lightbox">
  <img id="lbImg" src="" style="display:none" />
  <video id="lbVideo" controls style="display:none"></video>
  <div class="info" id="lbInfo"></div>
</div>
<div class="toast" id="toast"></div>

<script>
var RAW_DATA  = __BLEND_DATA__;
var BUCKET    = "__BUCKET__";
var TABLE     = "__TABLE__";
var BLEND_NUM = __BLEND_NUM__;

var flagged = new Set();

function updateCounter(){
  document.getElementById("counter").textContent = flagged.size + " flagged";
}
function toast(msg){
  var el = document.getElementById("toast");
  el.textContent = msg; el.classList.add("show");
  setTimeout(function(){ el.classList.remove("show"); }, 2500);
}

function toggle(blendId){
  flagged.has(blendId) ? flagged.delete(blendId) : flagged.add(blendId);
  var card = document.querySelector('[data-blend="' + CSS.escape(blendId) + '"]');
  if(card) card.classList.toggle("flagged", flagged.has(blendId));
  updateCounter();
}

function selectAll(){ getVisible().forEach(function(r){ flagged.add(r.blend_id); syncCard(r.blend_id); }); updateCounter(); }
function deselectAll(){ flagged.clear(); document.querySelectorAll(".blend-card.flagged").forEach(function(c){ c.classList.remove("flagged"); }); updateCounter(); }
function invertSelection(){ getVisible().forEach(function(r){ flagged.has(r.blend_id)?flagged.delete(r.blend_id):flagged.add(r.blend_id); syncCard(r.blend_id); }); updateCounter(); }
function syncCard(id){ var c=document.querySelector('[data-blend="' + CSS.escape(id) + '"]'); if(c) c.classList.toggle("flagged",flagged.has(id)); }

function getVisible(){
  var q = document.getElementById("search").value.toLowerCase();
  var fo = document.getElementById("showFlaggedOnly").checked;
  return RAW_DATA.filter(function(r){
    if(fo && !flagged.has(r.blend_id)) return false;
    if(q){
      var haystack = [r.blend_id, r.user_id, r.email, r.status].join(" ").toLowerCase();
      if(haystack.indexOf(q) === -1) return false;
    }
    return true;
  });
}

function sorted(items){
  var s = document.getElementById("sortBy").value;
  var copy = items.slice();
  if(s==="date_desc") copy.sort(function(a,b){ return (b.created_at||"").localeCompare(a.created_at||""); });
  if(s==="date_asc")  copy.sort(function(a,b){ return (a.created_at||"").localeCompare(b.created_at||""); });
  if(s==="status")    copy.sort(function(a,b){ return (a.status||"").localeCompare(b.status||""); });
  if(s==="user_id")   copy.sort(function(a,b){ return (a.user_id||"").localeCompare(b.user_id||""); });
  if(s==="blend_id")  copy.sort(function(a,b){ return a.blend_id.localeCompare(b.blend_id); });
  return copy;
}

function statusClass(s){
  if(s==="completed") return "status-completed";
  if(s==="failed" || s==="error") return "status-failed";
  return "status-pending";
}

var PAGE_SIZE = 20;
var currentPage = 0;
var currentItems = [];

var lazyObserver = new IntersectionObserver(function(entries){
  entries.forEach(function(entry){
    if(!entry.isIntersecting) return;
    var card = entry.target;
    card.querySelectorAll("img[data-src]").forEach(function(img){
      img.src = img.getAttribute("data-src");
      img.removeAttribute("data-src");
    });
    card.querySelectorAll("video[data-src]").forEach(function(vid){
      vid.src = vid.getAttribute("data-src");
      vid.removeAttribute("data-src");
    });
    lazyObserver.unobserve(card);
  });
}, { rootMargin: "200px" });

function buildCard(r){
    var card = document.createElement("div");
    card.className = "blend-card" + (flagged.has(r.blend_id) ? " flagged" : "");
    card.setAttribute("data-blend", r.blend_id);
    card.addEventListener("click", function(){ toggle(r.blend_id); });

    var assets = document.createElement("div");
    assets.className = "blend-assets";

    var srcCell = document.createElement("div");
    srcCell.className = "asset-cell";
    var srcLabel = document.createElement("span");
    srcLabel.className = "label";
    srcLabel.textContent = "Source";
    srcCell.appendChild(srcLabel);
    if(r.source_url){
      var srcImg = document.createElement("img");
      srcImg.setAttribute("data-src", r.source_url);
      srcImg.alt = "Source";
      srcImg.addEventListener("click", function(e){ e.stopPropagation(); openLightbox("img", r.source_url, "Source: " + r.source_key); });
      srcCell.appendChild(srcImg);
    } else {
      var noSrc = document.createElement("div");
      noSrc.className = "no-asset";
      noSrc.textContent = "No source image";
      srcCell.appendChild(noSrc);
    }
    assets.appendChild(srcCell);

    var rndCell = document.createElement("div");
    rndCell.className = "asset-cell";
    var rndLabel = document.createElement("span");
    rndLabel.className = "label";
    rndLabel.textContent = "Random";
    rndCell.appendChild(rndLabel);
    if(r.random_url){
      var rndImg = document.createElement("img");
      rndImg.setAttribute("data-src", r.random_url);
      rndImg.alt = "Random";
      rndImg.addEventListener("click", function(e){ e.stopPropagation(); openLightbox("img", r.random_url, "Random: " + r.random_key); });
      rndCell.appendChild(rndImg);
    } else {
      var noRnd = document.createElement("div");
      noRnd.className = "no-asset";
      noRnd.textContent = "No random image";
      rndCell.appendChild(noRnd);
    }
    assets.appendChild(rndCell);

    var vidCell = document.createElement("div");
    vidCell.className = "asset-cell";
    var vidLabel = document.createElement("span");
    vidLabel.className = "label";
    vidLabel.textContent = "Video";
    vidCell.appendChild(vidLabel);
    if(r.video_url){
      var vid = document.createElement("video");
      vid.setAttribute("data-src", r.video_url + "#t=0.1");
      vid.preload = "none";
      vid.muted = true;
      vid.addEventListener("click", function(e){ e.stopPropagation(); openLightbox("video", r.video_url, "Video: " + r.video_key); });
      vid.addEventListener("mouseenter", function(){ if(this.src) this.play(); });
      vid.addEventListener("mouseleave", function(){ this.pause(); this.currentTime = 0.1; });
      vidCell.appendChild(vid);
    } else {
      var noVid = document.createElement("div");
      noVid.className = "no-asset";
      noVid.textContent = "No video";
      vidCell.appendChild(noVid);
    }
    assets.appendChild(vidCell);

    card.appendChild(assets);

    var meta = document.createElement("div");
    meta.className = "blend-meta";

    var tagSpan = document.createElement("span");
    tagSpan.className = "tag";
    tagSpan.textContent = r.blend_id.substring(0, 8) + "...";
    meta.appendChild(tagSpan);

    var statusSpan = document.createElement("span");
    statusSpan.className = statusClass(r.status);
    statusSpan.textContent = r.status || "unknown";
    meta.appendChild(statusSpan);

    var dateSpan = document.createElement("span");
    dateSpan.textContent = r.created_at || "no date";
    meta.appendChild(dateSpan);

    var userSpan = document.createElement("span");
    userSpan.textContent = r.user_id ? r.user_id.substring(0, 8) + "..." : "no user";
    meta.appendChild(userSpan);

    if(r.email){
      var emailSpan = document.createElement("span");
      emailSpan.textContent = r.email;
      meta.appendChild(emailSpan);
    }

    card.appendChild(meta);
    return card;
}

function appendPage(){
  var gallery = document.getElementById("gallery");
  var start = currentPage * PAGE_SIZE;
  var end = Math.min(start + PAGE_SIZE, currentItems.length);

  for(var i = start; i < end; i++){
    var card = buildCard(currentItems[i]);
    gallery.appendChild(card);
    lazyObserver.observe(card);
  }
  currentPage++;

  var oldBtn = document.getElementById("loadMore");
  if(oldBtn) oldBtn.remove();

  if(currentPage * PAGE_SIZE < currentItems.length){
    var btn = document.createElement("button");
    btn.id = "loadMore";
    btn.textContent = "Load more (" + (currentItems.length - currentPage * PAGE_SIZE) + " remaining)";
    btn.style.cssText = "display:block;margin:20px auto;padding:10px 28px;border:1px solid #3a3a5a;border-radius:6px;background:#1e1e38;color:#ccc;cursor:pointer;font-size:.9rem";
    btn.addEventListener("click", function(){ appendPage(); });
    gallery.appendChild(btn);
  }
}

function renderGallery(){
  currentItems = sorted(getVisible());
  currentPage = 0;
  var gallery = document.getElementById("gallery");
  while(gallery.firstChild) gallery.removeChild(gallery.firstChild);

  appendPage();

  document.getElementById("stats").textContent =
    BUCKET + " · Blend " + BLEND_NUM + " · " + currentItems.length + " blend records";
}

function openLightbox(type, url, info){
  var imgEl = document.getElementById("lbImg");
  var vidEl = document.getElementById("lbVideo");
  imgEl.style.display = "none";
  vidEl.style.display = "none";

  if(type === "img"){
    imgEl.src = url;
    imgEl.style.display = "block";
  } else {
    vidEl.src = url;
    vidEl.style.display = "block";
    vidEl.play();
  }
  document.getElementById("lbInfo").textContent = info;
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox(){
  document.getElementById("lightbox").classList.remove("open");
  document.getElementById("lbImg").src = "";
  var v = document.getElementById("lbVideo");
  v.pause(); v.src = "";
}

document.getElementById("lightbox").addEventListener("click", closeLightbox);
document.addEventListener("keydown", function(e){ if(e.key==="Escape") closeLightbox(); });

document.getElementById("btnSelectAll").addEventListener("click", selectAll);
document.getElementById("btnDeselect").addEventListener("click", deselectAll);
document.getElementById("btnInvert").addEventListener("click", invertSelection);
document.getElementById("search").addEventListener("input", function(){ renderGallery(); });
document.getElementById("sortBy").addEventListener("change", function(){ renderGallery(); });
document.getElementById("showFlaggedOnly").addEventListener("change", function(){ renderGallery(); });

document.getElementById("btnExport").addEventListener("click", function(){
  if(flagged.size===0){ toast("Nothing flagged - select blends first"); return; }
  var manifestItems = [];
  flagged.forEach(function(blendId){
    var r = RAW_DATA.find(function(d){ return d.blend_id === blendId; });
    manifestItems.push({
      blend_id: blendId,
      source_image_key: r ? r.source_key : "",
      random_image_key: r ? r.random_key : "",
      s3_video_key: r ? r.video_key : "",
    });
  });
  var manifest = {
    created_at: new Date().toISOString(),
    blend_number: BLEND_NUM,
    bucket: BUCKET,
    table: TABLE,
    items: manifestItems,
  };
  var blob = new Blob([JSON.stringify(manifest, null, 2)], {type:"application/json"});
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "blend_manifest_blend" + BLEND_NUM + "_" + Date.now() + ".json";
  a.click();
  toast("Exported " + flagged.size + " blend(s) - send the downloaded JSON file to Yonatan");
});

renderGallery();
</script>
</body>
</html>"""


def generate_html(records: list[dict], bucket: str, table: str,
                  blend_num: int) -> str:
    data_json = json.dumps(records, default=str)
    html = HTML_TEMPLATE
    html = html.replace("__BLEND_DATA__", data_json)
    html = html.replace("__BUCKET__", bucket)
    html = html.replace("__TABLE__", table)
    html = html.replace("__BLEND_NUM__", str(blend_num))
    return html


# ---------------------------------------------------------------------------
# Callable entry point (used by launcher)
# ---------------------------------------------------------------------------

def run_review_blends(blend_number: int, on_progress=None,
                      url_expiry: int = 86400) -> Path:
    cfg = get_blend_config(blend_number)
    bucket = cfg["bucket"]
    table = cfg["table"]

    _emit(f"{'='*60}", on_progress)
    _emit(f"  Blend {blend_number}:  bucket={bucket}  table={table}", on_progress)
    _emit(f"{'='*60}\n", on_progress)

    s3 = get_s3_client()
    dynamodb = get_dynamodb_resource()

    items = scan_dynamodb(dynamodb, table, on_progress)
    if not items:
        _emit("No blend records found. Nothing to review.", on_progress)
        return None

    records = build_blend_records(s3, bucket, items, expiry=url_expiry,
                                 on_progress=on_progress)

    completed = sum(1 for r in records if r["status"] == "completed")
    _emit(f"\n  {len(records)} blend records total", on_progress)
    _emit(f"  {completed} completed", on_progress)
    _emit(f"  {len(records) - completed} other statuses\n", on_progress)

    html = generate_html(records, bucket, table, blend_number)

    out_dir = app_data_dir()
    out_path = out_dir / f"review_blends{blend_number}.html"
    out_path.write_text(html, encoding="utf-8")
    _emit(f"Gallery saved to: {out_path.resolve()}", on_progress)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate a blend-review gallery from DynamoDB")
    p.add_argument("--blend", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="Blend number (1-5, default 1)")
    p.add_argument("--output", default=None,
                   help="Output HTML path (default: review_blends<N>.html)")
    p.add_argument("--no-open", action="store_true",
                   help="Don't auto-open the HTML in a browser")
    p.add_argument("--url-expiry", type=int, default=86400,
                   help="Presigned URL expiry in seconds (default: 86400 = 24 h)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = run_review_blends(args.blend, url_expiry=args.url_expiry)

    if out_path and not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
        print("Opened in browser.")

    print("\nWorkflow:")
    print("  1. Review blends - each card shows source image, random image, and video")
    print("  2. Click a card to flag it (click images/video for full-size view)")
    print("  3. Click 'Export Deletion Manifest' to download the JSON")
    print("  4. Send the downloaded JSON file to Yonatan for deletion")


if __name__ == "__main__":
    main()
