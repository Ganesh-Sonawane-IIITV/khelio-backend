"""
Google AI Studio only — single-file product imagery pipeline.

What it does:
1) Input: YouTube URL (hardcoded in __main__, or call process_youtube_video(url))
2) Extract key frames (≈1 fps)
3) Use Gemini (Google AI Studio) to:
   - Identify products + pick best frame per product (JSON with bbox)
   - Segment product via polygon (normalized points)
   - Robust JSON parsing + graceful fallbacks
4) Enhance: generate 2–3 “background-only” images using Google Images API (Imagen 3) if available,
   else try Gemini image output, else procedural gradient backgrounds
5) Composite product cutout on backgrounds
6) Optionally saves artifacts to disk (save_dir="out") and writes result.json

Install:
  pip install yt_dlp opencv-python pillow numpy requests google-generativeai

Environment:
  GOOGLE_AI_API_KEY=<your_google_ai_studio_key>
  (optional) GEMINI_ANALYSIS_MODEL=gemini-1.5-flash
  (optional) GEMINI_SEGMENT_MODEL=gemini-1.5-flash
  (optional) GEMINI_IMAGE_MODEL=gemini-2.0-flash-exp (fallback if Imagen unavailable)
"""

import os
import io
import json
import base64
import random
import string
from typing import Any, Dict, List, Tuple, Optional

import requests
import yt_dlp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import google.generativeai as genai

# ----------------------------- Config -----------------------------
API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
if not API_KEY:
    print("[warn] GOOGLE_AI_API_KEY not set. The pipeline will try local fallbacks (heuristics/procedural).")

genai.configure(api_key=API_KEY or "invalid")

GEMINI_ANALYSIS_MODEL = os.getenv("GEMINI_ANALYSIS_MODEL", "gemini-1.5-flash")  # reliable, multimodal
GEMINI_SEGMENT_MODEL  = os.getenv("GEMINI_SEGMENT_MODEL",  "gemini-1.5-flash")
GEMINI_IMAGE_MODEL    = os.getenv("GEMINI_IMAGE_MODEL",    "gemini-2.0-flash-exp")  # may output inline images in some accounts

# Attempt to use Images API (Imagen 3). SDK availability varies by account/version.
IMAGEN_MODEL_ID = os.getenv("IMAGEN_MODEL_ID", "imagen-3.0-generate")

# Frame sampling
FRAMES_PER_SECOND = 1
MAX_TOTAL_FRAMES = 90

# ----------------------------- Utils -----------------------------
def random_id(n=6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def np_bgr_to_jpg_b64(arr: np.ndarray, quality: int = 90) -> str:
    arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", arr_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf).decode("utf-8")

def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return base64.b64encode(bio.getvalue()).decode("utf-8")

def safe_name(name: str, max_len: int = 60) -> str:
    import re
    return re.sub(r"[^\w\-_]+", "_", name)[:max_len] or "item"

def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

# ----------------------------- YouTube + Frames -----------------------------
def download_youtube(youtube_url: str) -> str:
    ydl_opts = {
        "format": "best[height<=720]",
        "outtmpl": f"dl_{random_id()}_%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return ydl.prepare_filename(info)

def extract_key_frames(video_path: str, fps_sample: int = FRAMES_PER_SECOND, max_frames: int = MAX_TOTAL_FRAMES) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = int(fps) if fps > 0 else 30
    interval = max(int(fps / max(1, fps_sample)), 1)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            frames.append(np_bgr_to_jpg_b64(frame))
        idx += 1
    cap.release()
    return frames

# ----------------------------- Gemini helpers -----------------------------
def gemini_generate_json(parts: List[Any], schema: Dict[str, Any], model: str, temperature: float = 0.2, timeout: int = 120) -> Optional[Dict[str, Any]]:
    """
    Uses Google AI Studio SDK to enforce JSON via response_schema.
    Returns dict or None on failure.
    """
    try:
        mdl = genai.GenerativeModel(model)
        cfg = {"temperature": temperature, "response_mime_type": "application/json", "response_schema": schema}
        resp = mdl.generate_content(parts, generation_config=cfg, request_options={"timeout": timeout})
        # Preferred: resp.text is already JSON string (thanks to response_mime_type)
        return json.loads(resp.text)
    except Exception as e:
        print(f"[gemini_json:{model}] {e}")
        # Try to parse best-effort JSON from any text content
        try:
            text = ""
            if 'resp' in locals():
                # Collect any text parts
                for cand in resp.candidates or []:
                    for part in cand.content.parts or []:
                        if getattr(part, "text", None):
                            text += part.text
            if text:
                obj = _parse_json_safely(text)
                return obj
        except Exception as _:
            pass
    return None

def _parse_json_safely(text: str) -> Dict[str, Any]:
    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced blocks
    for fence in ("```json", "```JSON", "```"):
        if fence in text:
            try:
                chunk = text.split(fence, 1)[1]
                chunk = chunk.split("```", 1)[0]
                return json.loads(chunk.strip())
            except Exception:
                pass
    # Brace matching
    obj = _extract_first_json_object(text)
    if obj:
        return json.loads(obj)
    raise RuntimeError("Failed to parse JSON")

def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None

def try_imagen_background(prompt: str, aspect_ratio: str = "1:1") -> Optional[str]:
    """
    Try Google Images API (Imagen 3) via python SDK if available to your account.
    Returns base64 string or None.
    """
    try:
        ImageGenerationModel = getattr(genai, "ImageGenerationModel", None)
        if ImageGenerationModel is None:
            return None
        img_model = ImageGenerationModel(IMAGEN_MODEL_ID)
        result = img_model.generate_images(
            prompt=f"Background-only image. No text or logos. {prompt}",
            number_of_images=1,
            aspect_ratio=aspect_ratio
        )
        if result and getattr(result, "images", None):
            img0 = result.images[0]
            if hasattr(img0, "image_bytes"):
                return base64.b64encode(img0.image_bytes).decode("utf-8")
            if hasattr(img0, "inline_data") and getattr(img0.inline_data, "data", None):
                return img0.inline_data.data
        return None
    except Exception as e:
        print(f"[imagen] {e}")
        return None

def try_gemini_image(prompt: str, model: str = GEMINI_IMAGE_MODEL, timeout: int = 120) -> Optional[str]:
    """
    Try Gemini model that can emit inline image (experimental/if available).
    """
    try:
        mdl = genai.GenerativeModel(model)
        resp = mdl.generate_content([{"text": prompt}], request_options={"timeout": timeout})
        for cand in resp.candidates or []:
            for part in cand.content.parts or []:
                if getattr(part, "inline_data", None):
                    return part.inline_data.data
    except Exception as e:
        print(f"[gemini_image:{model}] {e}")
    return None

# ----------------------------- Identification -----------------------------
def identify_products_and_best_frames(frames_b64: List[str], batch_size: int = 3) -> List[Dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "best_frame_index": {"type": "integer"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "x":{"type":"number"},"y":{"type":"number"},
                                "w":{"type":"number"},"h":{"type":"number"}
                            },
                            "required": ["x","y","w","h"]
                        }
                    },
                    "required": ["name","best_frame_index"]
                }
            }
        },
        "required": ["products"]
    }
    prompt = (
        "Analyze these frames and identify distinct product(s) shown. "
        "For each product, return the best frame index relative to THIS BATCH (0..N-1), "
        "a short reason, a confidence 0..1, and a tight bbox (normalized x,y,w,h in 0..1). "
        "Return strictly valid JSON matching the schema."
    )

    found: List[Dict[str, Any]] = []
    for i in range(0, len(frames_b64), batch_size):
        batch = frames_b64[i:i+batch_size]
        parts = [{"text": prompt}]
        for b in batch:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b}})
        out = gemini_generate_json(parts, schema, model=GEMINI_ANALYSIS_MODEL, temperature=0.1)
        if not out:
            print(f"[identify] Skipping batch {i//batch_size+1}: no structured response")
            continue
        for p in out.get("products", []):
            bbox = p.get("bbox", {}) or {}
            bbox = {
                "x": clamp01(bbox.get("x", 0.2)),
                "y": clamp01(bbox.get("y", 0.2)),
                "w": clamp01(bbox.get("w", 0.6)),
                "h": clamp01(bbox.get("h", 0.6)),
            }
            local = int(p.get("best_frame_index", 0))
            global_idx = min(i + local, len(frames_b64) - 1)
            p["best_frame_index"] = global_idx
            p["bbox"] = bbox
            p["frame_b64"] = frames_b64[global_idx]
            try:
                p["confidence"] = float(p.get("confidence", 0.0))
            except Exception:
                p["confidence"] = 0.0
            found.append(p)

    # Merge duplicates by name (highest confidence)
    merged = {}
    for p in found:
        key = (p.get("name") or f"product_{random_id()}").strip().lower()
        if key not in merged or (p.get("confidence", 0.0) > merged[key].get("confidence", 0.0)):
            merged[key] = p

    if not merged:
        # Heuristic fallback: pick sharpest 1-2 frames
        scores = []
        for idx, fb in enumerate(frames_b64):
            try:
                g = b64_to_pil(fb).convert("L")
                s = laplacian_sharpness(np.array(g))
            except Exception:
                s = 0.0
            scores.append((s, idx))
        scores.sort(reverse=True)
        chosen = [idx for _, idx in scores[:max(1, min(2, len(scores)))]]
        return [{
            "name": f"Product_{n+1}",
            "best_frame_index": gi,
            "reason": "Sharpest frame heuristic",
            "confidence": 0.0,
            "bbox": {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
            "frame_b64": frames_b64[gi],
        } for n, gi in enumerate(chosen)]

    return list(merged.values())

# ----------------------------- Segmentation -----------------------------
def segment_product_polygon(product_name: str, frame_b64: str) -> Optional[Dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "polygon": {
                "type": "array",
                "minItems": 3,
                "items": {"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}}, "required":["x","y"]}
            },
            "tight_bbox": {
                "type": "object",
                "properties": {"x":{"type":"number"},"y":{"type":"number"},"w":{"type":"number"},"h":{"type":"number"}},
                "required": ["x","y","w","h"]
            }
        },
        "required": ["polygon","tight_bbox"]
    }
    prompt = (
        f"Segment the product '{product_name}' precisely. "
        "Return a polygon (list of normalized points (x,y) in 0..1) tracing the outer contour, "
        "and a tight bounding box (x,y,w,h) in 0..1. "
        "Return strictly valid JSON matching the schema."
    )
    parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}}]
    out = gemini_generate_json(parts, schema, model=GEMINI_SEGMENT_MODEL, temperature=0.1)
    if not out:
        return None
    # Clean/clamp
    poly = out.get("polygon", []) or []
    cleaned = [{"x": clamp01(pt.get("x", 0.5)), "y": clamp01(pt.get("y", 0.5))} for pt in poly if isinstance(pt, dict)]
    bbox = out.get("tight_bbox", {}) or {}
    bbox = {"x": clamp01(bbox.get("x", 0.2)), "y": clamp01(bbox.get("y", 0.2)), "w": clamp01(bbox.get("w", 0.6)), "h": clamp01(bbox.get("h", 0.6))}
    return {"polygon": cleaned, "tight_bbox": bbox}

def build_mask_and_crop(frame_b64: str, polygon: List[Dict[str, float]]) -> Tuple[str, str]:
    img = b64_to_pil(frame_b64).convert("RGBA")
    w, h = img.size
    pts = [(int(pt["x"] * w), int(pt["y"] * h)) for pt in polygon] if polygon and len(polygon) >= 3 else [(0,0),(w-1,0),(w-1,h-1),(0,h-1)]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    rgba = img.copy()
    rgba.putalpha(mask)
    bbox = mask.getbbox() or (0, 0, w, h)
    crop = rgba.crop(bbox)
    return pil_to_b64(mask, "PNG"), pil_to_b64(crop, "PNG")

def grabcut_cutout_from_bbox(frame_b64: str, bbox: Dict[str,float], iters: int = 5) -> Tuple[str,str]:
    img_rgba = b64_to_pil(frame_b64).convert("RGBA")
    img_bgr = cv2.cvtColor(np.array(img_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    x = int(clamp01(bbox.get("x",0.2)) * w)
    y = int(clamp01(bbox.get("y",0.2)) * h)
    bw = int(clamp01(bbox.get("w",0.6)) * w)
    bh = int(clamp01(bbox.get("h",0.6)) * h)
    # rect expects (x, y, width, height)
    rect = (x, y, max(2, bw), max(2, bh))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask==1) | (mask==3), 255, 0).astype(np.uint8)
    except Exception:
        fg_mask = np.zeros((h, w), np.uint8)
        fg_mask[y:y+bh, x:x+bw] = 255

    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = fg_mask

    ys, xs = np.where(fg_mask > 0)
    if len(xs)==0 or len(ys)==0:
        x1, y1, x2, y2 = x, y, x+bw, y+bh
    else:
        x1, x2 = max(0, xs.min()), min(w, xs.max()+1)
        y1, y2 = max(0, ys.min()), min(h, ys.max()+1)

    crop = rgba[y1:y2, x1:x2]
    mask_crop = fg_mask[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop)
    mask_pil = Image.fromarray(mask_crop, mode="L")
    return pil_to_b64(mask_pil, "PNG"), pil_to_b64(crop_pil, "PNG")

# ----------------------------- Enhancement -----------------------------
def procedural_background(size=(1024,1024), style="studio") -> Image.Image:
    w, h = size
    if style == "studio":
        # light sweep gradient
        img = Image.new("RGB", size, (250,250,250))
        overlay = Image.new("L", size, 0)
        d = ImageDraw.Draw(overlay)
        d.ellipse([int(-0.2*w), int(-0.2*h), int(1.2*w), int(0.9*h)], fill=220)
        img = Image.composite(Image.new("RGB", size, (255,255,255)), img, overlay)
        return img
    elif style == "lifestyle":
        # subtle wood-like gradient bands
        base = Image.new("RGB", size, (230, 220, 205))
        for y in range(h):
            tone = int(10 * np.sin(y / 30.0))  # subtle bands
            for x in range(w):
                r,g,b = base.getpixel((x,y))
                base.putpixel((x,y), (min(255, r+tone), min(255, g+tone), min(255, b+tone)))
        base = base.filter(ImageFilter.GaussianBlur(radius=1.2))
        return base
    else:
        # creative gradient
        img = Image.new("RGB", size, (30, 30, 40))
        overlay = Image.new("L", size, 0)
        d = ImageDraw.Draw(overlay)
        d.ellipse([int(-0.3*w), int(-0.2*h), int(1.1*w), int(1.2*h)], fill=180)
        img = Image.composite(Image.new("RGB", size, (80,20,120)), img, overlay)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        return img

def composite_center(bg: Image.Image, product_rgba: Image.Image) -> Image.Image:
    bg = bg.convert("RGBA")
    bw, bh = bg.size
    pw, ph = product_rgba.size
    target_w = int(bw * 0.6)
    scale = target_w / max(pw, 1)
    new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
    prod = product_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Soft shadow
    shadow = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    sh_mask = Image.new("L", (new_w, new_h), 0)
    d = ImageDraw.Draw(sh_mask)
    ellipse_h = max(4, int(new_h * 0.15))
    d.ellipse([new_w*0.1, new_h-ellipse_h, new_w*0.9, new_h], fill=120)
    sh_mask = sh_mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(new_w * 0.02))))
    shadow.putalpha(sh_mask)

    composed = bg.copy()
    x = (bw - new_w) // 2
    y = (bh - new_h) // 2
    composed.alpha_composite(shadow, (x, y + int(new_h * 0.04)))
    composed.alpha_composite(prod, (x, y))
    return composed

def generate_background_b64(style_prompt: str) -> str:
    """
    Try Imagen 3 via SDK; then Gemini image output; else procedural.
    Returns base64 PNG.
    """
    # Prefer Imagen 3 (if available)
    b64 = try_imagen_background(style_prompt, aspect_ratio="1:1")
    if not b64:
        # Try Gemini experimental image output
        b64 = try_gemini_image(f"Background-only image. No text or logos. {style_prompt}", model=GEMINI_IMAGE_MODEL)
    if b64:
        return b64

    # Procedural fallback (always works)
    size = (1024, 1024)
    if "white sweep" in style_prompt.lower():
        bg = procedural_background(size, "studio")
    elif "wood" in style_prompt.lower() or "lifestyle" in style_prompt.lower():
        bg = procedural_background(size, "lifestyle")
    else:
        bg = procedural_background(size, "creative")
    return pil_to_b64(bg, "PNG")

def enhance_product_shots(product_name: str, product_crop_b64: str) -> List[str]:
    styles = [
        "High-end studio white sweep background, soft even lighting, photorealistic, 1:1, 4k.",
        "Warm lifestyle background with subtle light wood texture, natural light, 1:1, photorealistic.",
        "Premium artistic gradient background with bold contrast and vignette, 1:1, editorial style."
    ]
    prod_rgba = b64_to_pil(product_crop_b64).convert("RGBA")
    shots = []
    for style in styles:
        bg_b64 = generate_background_b64(style)
        try:
            bg = b64_to_pil(bg_b64).convert("RGBA")
        except Exception:
            # As a last resort, generate procedural again directly
            if "white sweep" in style.lower():
                bg = procedural_background((1024,1024), "studio").convert("RGBA")
            elif "wood" in style.lower():
                bg = procedural_background((1024,1024), "lifestyle").convert("RGBA")
            else:
                bg = procedural_background((1024,1024), "creative").convert("RGBA")
        bg = bg.resize((1024, 1024), Image.LANCZOS)
        composed = composite_center(bg, prod_rgba)
        shots.append(pil_to_b64(composed, "PNG"))
    return shots

# ----------------------------- Pipeline -----------------------------
def process_youtube_video(youtube_url: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
    # 1) Download + frames
    video_path = download_youtube(youtube_url)
    try:
        frames_b64 = extract_key_frames(video_path, fps_sample=FRAMES_PER_SECOND, max_frames=MAX_TOTAL_FRAMES)
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
    if not frames_b64:
        raise RuntimeError("No frames extracted from video.")

    # 2) Identify products + best frame
    products = identify_products_and_best_frames(frames_b64, batch_size=3)

    # 3) Segmentation (polygon via Gemini; fallback GrabCut on bbox; last-resort rectangle)
    for p in products:
        name = p.get("name", "Product")
        frame_b64 = p["frame_b64"]
        bbox = p.get("bbox", {"x":0.2,"y":0.2,"w":0.6,"h":0.6})
        seg = segment_product_polygon(name, frame_b64)
        if seg and seg.get("polygon"):
            mask_b64, crop_b64 = build_mask_and_crop(frame_b64, seg["polygon"])
            p["segmentation"] = {
                "polygon": seg["polygon"],
                "tight_bbox": seg.get("tight_bbox", bbox),
                "mask_b64": mask_b64,
                "cropped_b64": crop_b64
            }
        else:
            # High-quality local fallback
            try:
                mask_b64, crop_b64 = grabcut_cutout_from_bbox(frame_b64, bbox)
            except Exception:
                # Rectangular fallback
                mask_b64, crop_b64 = _rect_crop(frame_b64, bbox)
            p["segmentation"] = {
                "polygon": [],
                "tight_bbox": bbox,
                "mask_b64": mask_b64,
                "cropped_b64": crop_b64
            }

    # 4) Enhancement (Imagen/Gemini/procedural) + composite
    for p in products:
        crop_b64 = p["segmentation"]["cropped_b64"]
        p["enhanced"] = enhance_product_shots(p["name"], crop_b64)

    # Optional save
    if save_dir:
        ensure_dir(save_dir)
        for i, p in enumerate(products):
            pd = os.path.join(save_dir, f"{i:02d}_{safe_name(p['name'])}")
            ensure_dir(pd)
            try: b64_to_pil(p["frame_b64"]).convert("RGB").save(os.path.join(pd, "01_best_frame.jpg"), "JPEG", quality=95)
            except Exception: pass
            try: b64_to_pil(p["segmentation"]["mask_b64"]).save(os.path.join(pd, "02_mask.png"), "PNG")
            except Exception: pass
            try: b64_to_pil(p["segmentation"]["cropped_b64"]).save(os.path.join(pd, "03_product_rgba.png"), "PNG")
            except Exception: pass
            for j, eb64 in enumerate(p.get("enhanced", [])):
                try: b64_to_pil(eb64).save(os.path.join(pd, f"1{j+1}_enhanced.png"), "PNG")
                except Exception: pass
        # Save full JSON (for frontend)
        with open(os.path.join(save_dir, "result.json"), "w") as f:
            json.dump({
                "youtube_url": youtube_url,
                "products": products,
                "workflow_docs": {
                    "analysis": {
                        "model": GEMINI_ANALYSIS_MODEL,
                        "notes": "Gemini JSON with response_schema; fallback heuristic if unavailable"
                    },
                    "segmentation": {
                        "model": GEMINI_SEGMENT_MODEL,
                        "approach": "Polygon (normalized) -> raster mask -> RGBA cutout; fallback GrabCut on bbox"
                    },
                    "enhancement": {
                        "models": {
                            "preferred": IMAGEN_MODEL_ID,
                            "fallback": GEMINI_IMAGE_MODEL,
                            "final": "procedural gradients"
                        },
                        "approach": "Generate background-only images, then center-composite product with soft shadow",
                        "styles": ["studio white", "lifestyle wood", "creative gradient"]
                    }
                }
            }, f, indent=2)

    return {
        "youtube_url": youtube_url,
        "products": products
    }

def _rect_crop(frame_b64: str, bbox: Dict[str, float]) -> Tuple[str, str]:
    img = b64_to_pil(frame_b64).convert("RGBA")
    w, h = img.size
    x1 = int(clamp01(bbox.get("x", 0.2)) * w)
    y1 = int(clamp01(bbox.get("y", 0.2)) * h)
    x2 = int((clamp01(bbox.get("x", 0.2)) + clamp01(bbox.get("w", 0.6))) * w)
    y2 = int((clamp01(bbox.get("y", 0.2)) + clamp01(bbox.get("h", 0.6))) * h)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)
    rgba = img.copy()
    rgba.putalpha(mask)
    crop = rgba.crop((x1, y1, x2, y2))
    return pil_to_b64(mask, "PNG"), pil_to_b64(crop, "PNG")

# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    # Put your YT URL here or call process_youtube_video() from your code
    YT_URL = "https://www.youtube.com/watch?v=3z0_GlI47Nw"
    out = process_youtube_video(YT_URL, save_dir="out")

    # Compact summary (avoid printing huge base64)
    print("=== Summary ===")
    print(f"Video: {out['youtube_url']}")
    print(f"Products: {len(out['products'])}")
    for i, p in enumerate(out["products"], 1):
        print(f"- {i}. {p.get('name')} | best_frame: {p.get('best_frame_index')} | enhanced_shots: {len(p.get('enhanced', []))}")
    print("Artifacts saved to ./out (images) and out/result.json")