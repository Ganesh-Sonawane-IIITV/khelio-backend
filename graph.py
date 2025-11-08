# graph.py
import os
import json
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END

from back import (
    download_youtube,
    extract_key_frames,
    identify_products_and_best_frames,
    segment_product_polygon,
    build_mask_and_crop,
    grabcut_cutout_from_bbox,
    enhance_product_shots,
    ensure_dir,
    safe_name,
    b64_to_pil,
    _rect_crop,
)

class PipelineState(TypedDict, total=False):
    youtube_url: str
    frames_b64: List[str]
    products: List[Dict[str, Any]]
    save_dir: Optional[str]

def node_extract(state: PipelineState) -> Dict[str, Any]:
    url = state["youtube_url"]
    video_path = download_youtube(url)
    try:
        frames = extract_key_frames(video_path)
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
    return {"frames_b64": frames}

def node_identify(state: PipelineState) -> Dict[str, Any]:
    products = identify_products_and_best_frames(state["frames_b64"], batch_size=3)
    return {"products": products}

def node_segment(state: PipelineState) -> Dict[str, Any]:
    updated = []
    for p in state["products"]:
        name = p.get("name", "Product")
        frame_b64 = p["frame_b64"]
        bbox = p.get("bbox", {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6})
        seg = segment_product_polygon(name, frame_b64)
        if seg and seg.get("polygon"):
            mask_b64, crop_b64 = build_mask_and_crop(frame_b64, seg["polygon"])
            p["segmentation"] = {
                "polygon": seg["polygon"],
                "tight_bbox": seg.get("tight_bbox", bbox),
                "mask_b64": mask_b64,
                "cropped_b64": crop_b64,
            }
        else:
            try:
                mask_b64, crop_b64 = grabcut_cutout_from_bbox(frame_b64, bbox)
            except Exception:
                mask_b64, crop_b64 = _rect_crop(frame_b64, bbox)
            p["segmentation"] = {
                "polygon": [],
                "tight_bbox": bbox,
                "mask_b64": mask_b64,
                "cropped_b64": crop_b64,
            }
        updated.append(p)
    return {"products": updated}

def node_enhance(state: PipelineState) -> Dict[str, Any]:
    for p in state["products"]:
        p["enhanced"] = enhance_product_shots(p["name"], p["segmentation"]["cropped_b64"])
    return {"products": state["products"]}

def node_persist(state: PipelineState) -> Dict[str, Any]:
    save_dir = state.get("save_dir")
    if not save_dir:
        return {}
    ensure_dir(save_dir)
    products = state["products"]

    for i, p in enumerate(products):
        pd = os.path.join(save_dir, f"{i:02d}_{safe_name(p.get('name','item'))}")
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

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(
            {
                "youtube_url": state["youtube_url"],
                "products": products,
                "graph_nodes": ["extract", "identify", "segment", "enhance", "persist"],
            },
            f,
            indent=2,
        )
    return {}

def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("extract", node_extract)
    builder.add_node("identify", node_identify)
    builder.add_node("segment", node_segment)
    builder.add_node("enhance", node_enhance)
    builder.add_node("persist", node_persist)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "identify")
    builder.add_edge("identify", "segment")
    builder.add_edge("segment", "enhance")
    builder.add_edge("enhance", "persist")
    builder.add_edge("persist", END)

    return builder.compile()