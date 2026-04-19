"""
Automated mode
python /root/autodl-tmp/try_text_to_img3.py \
  --images_dir      "/root/autodl-tmp/flair_train_val/aerial/Z21_NF" \
  --model_path      "/root/autodl-tmp/models/HySCDG_Inpainting" \
  --controlnet_path "/root/autodl-tmp/models/HySCDG_ControlNet" \
  --save_dir        "/root/autodl-tmp/result_Englilsh/Z21_NF" \
  --api_key         "……" \
  --n_per_image     3 \
  --max_samples     600


Batch generate commands
python /root/autodl-tmp/try_text_to_img3.py \
  --jsonl_path      "/root/autodl-tmp/result_Englilsh/Z26_AA/auto_instructions.jsonl" \
  --model_path      "/root/autodl-tmp/models/HySCDG_Inpainting" \
  --controlnet_path "/root/autodl-tmp/models/HySCDG_ControlNet" \
  --save_dir        "/root/autodl-tmp/result_Englilsh/Z26_AA" \
  --api_key         "……" \
  --max_samples     200

"""

import os, sys, json, argparse, base64, io
import numpy as np
import torch
import rasterio
import pandas as pd
from PIL import Image
from openai import OpenAI
from scipy.ndimage import binary_dilation
from scipy.ndimage.measurements import label as connexLabel

# ========== HySCDG ==========
HYSCDG_ROOT = "/root/autodl-tmp/HySCDG"
sys.path.insert(0, HYSCDG_ROOT)
from auto_genetae_text import generate_instructions
from generation import diffusion
from src import flair
from src.utils import convert_to_color

# ========== config ==========
QWEN_BASE_URL  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TEXT_MODEL = "qwen-plus"       # Plain text: Parse source/target classes
QWEN_VL_MODEL   = "qwen-vl-plus"   # Visual: Locate target region bbox

CLASS_NAMES = {
    0:"unknown", 1:"building", 2:"pervious surface", 3:"impervious surface",
    4:"bare soil", 5:"water", 6:"coniferous", 7:"deciduous", 8:"brushwood",
    9:"vineyard", 10:"herbaceous vegetation", 11:"agricultural land",
    12:"plowed land", 13:"swimming pool", 14:"snow", 15:"clear cut",
    16:"mixed", 17:"ligneous", 18:"greenhouse", 19:"other"
}

LUT_COLORS = {
    1:(219,14,154), 2:(147,142,123), 3:(248,12,0),  4:(169,113,1),
    5:(21,83,174),  6:(25,74,38),    7:(70,228,131), 8:(243,166,13),
    9:(102,0,130),  10:(85,255,0),   11:(255,243,13),12:(228,223,124),
    13:(61,230,235),14:(255,255,255),15:(138,179,160),16:(107,113,79),
    17:(197,220,66),18:(153,153,255),19:(0,0,0),
}

# Dataset frequency (classes 0-19, from flair.py)
DATASET_FREQ = np.array([0., 8.14, 8.25, 13.72, 3.47, 4.88, 2.74, 15.38,
                          6.95, 3.13, 17.84, 10.98, 3.88, 0., 0., 0., 0., 0., 0., 0.])
DATASET_FREQ = DATASET_FREQ / (DATASET_FREQ.sum() + 1e-8)
VALID_REPLACE_CLASSES = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12]  


def pick_target_class_by_prob(ctrl: np.ndarray, source_classes: list) -> int:
    """
    Disappearing class handling: Select target class using the probabilistic method
    Probability ∝ Dataset Frequency / (Current Image Frequency + 1)
    Prioritize classes that are rare in the current image but common in the dataset (for semantic plausibility).
    Also exclude the source class itself.
    """
    H, W = ctrl.shape
    total = H * W

    # Class frequencies in the current image
    img_freq = np.zeros(20, dtype=np.float32)
    for c in range(20):
        img_freq[c] = (ctrl == c).sum() / total

    # Calculate probability: Dataset Frequency / (Image Frequency + epsilon)
    probs = np.zeros(20, dtype=np.float32)
    for c in VALID_REPLACE_CLASSES:
        if c not in source_classes:
            probs[c] = DATASET_FREQ[c] / (img_freq[c] + 0.01)

    if probs.sum() == 0:
        # fallback：Randomly select from valid classes, excluding the source class
        candidates = [c for c in VALID_REPLACE_CLASSES if c not in source_classes]
        return int(np.random.choice(candidates))

    probs = probs / probs.sum()
    chosen = int(np.random.choice(20, p=probs))
    print(f"[PROB] Disappearing class target selection: {chosen}={CLASS_NAMES.get(chosen)} "
          f"(Dataset freq={DATASET_FREQ[chosen]*100:.1f}%, Image freq={img_freq[chosen]*100:.1f}%)")
    return chosen


# Step0-A: Text model parses source/target classes
def parse_classes(instruction: str, api_key: str) -> dict:
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
    system = """You are a remote sensing image change analysis expert. Extract change information from user instructions and output JSON.

FLAIR class IDs:
1=building, 2=pervious surface, 3=impervious surface, 4=bare soil,
5=water, 6=coniferous, 7=deciduous, 8=brushwood, 9=vineyard,
10=herbaceous vegetation, 11=agricultural land, 12=plowed land, 19=other

Class mapping (instruction keywords → class IDs):
forest/trees/woodland → [6,7]
vegetation/grass/meadow → [10]
shrubs/brushwood/bushes → [8]
pervious surface/permeable ground/gravel → [2]
building/house/rooftop/residential → [1]
road/pavement/asphalt/street → [3]
water/river/lake/pond → [5]
farmland/agricultural land/crops/plowed land → [11,12]
bare soil/bare land/dirt → [4]
vineyard/grape field → [9]
snow → [14]
swimming pool/pool → [13]

Position extraction (keep original wording):
top-left / bottom-right / center / upper / lower / left / right / entire image, etc.

Instruction type:
- Replacement: explicitly says "change to / replace with / convert to" → fill target_class with corresponding ID
- Disappearance: says "remove / disappear / delete / clear" with no target → fill target_class with null

Output JSON only:
{
  "source_classes": [list of source class IDs],
  "target_class": target class ID or null,
  "position": "position words extracted from instruction, use 'entire image' if none",
  "source_desc": "source class name in English",
  "target_desc": "target class name in English, use 'auto-selected' for disappearance type"
}"""
    print(f"[Text] parsing: {instruction}")
    resp = client.chat.completions.create(
        model=QWEN_TEXT_MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":instruction}],
        temperature=0.1
    )
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    print(f"[Text] results: {raw}")
    return json.loads(raw)


# Step0-B:VL model locates target regions in the image 
def locate_region_with_vl(image_path: str, instruction: str,
                           source_desc: str, api_key: str, H: int, W: int) -> list:
    """
Use Qwen-VL to analyze the remote sensing image and return the pixel bounding box [x1, y1, x2, y2] of the target region.
The bbox uses the image coordinate system: x represents the column direction (0~W), and y represents the row direction (0~H).
    """
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    # tif to png for VL
    with rasterio.open(image_path) as src:
        img_np = src.read()[:3]  # (3,H,W)
    img_pil = Image.fromarray(np.moveaxis(img_np, 0, -1).astype(np.uint8))

    img_pil_resized = img_pil.resize((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    img_pil_resized.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()

    prompt = f'''This is an aerial remote sensing image (512x512 pixels, origin at top-left, x increases rightward, y increases downward).
User instruction: "{instruction}"
Please locate the target object ({source_desc}) described in the instruction, paying attention to position, quantity and all other details.
Output JSON only: {{"bbox_2d": [x1, y1, x2, y2]}}\"\"\"'''

    print(f"[LLM-VL] locating: {source_desc}...")
    try:
        resp = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            temperature=0.1
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        print(f"[LLM-VL] return: {raw}")

        # bbox
        parsed = json.loads(raw)
        bbox_512 = parsed["bbox_2d"]  # [x1,y1,x2,y2] in 512x512 space

        # Map back to original image coordinates
        scale_x = W / 512.0
        scale_y = H / 512.0
        x1 = int(bbox_512[0] * scale_x)
        y1 = int(bbox_512[1] * scale_y)
        x2 = int(bbox_512[2] * scale_x)
        y2 = int(bbox_512[3] * scale_y)

        # Ensure coordinates are valid
        x1, x2 = max(0, min(x1, W-1)), max(0, min(x2, W-1))
        y1, y2 = max(0, min(y1, H-1)), max(0, min(y2, H-1))
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        print(f"[LLM-VL] Localization result (original image coords): x[{x1}:{x2}] y[{y1}:{y2}]")
        print(f"[LLM-VL] Region size: {x2-x1} x {y2-y1} pixels")
        return [x1, y1, x2, y2]  # [col_min, row_min, col_max, row_max]

    except Exception as e:
        print(f"[LLM-VL] Visual localization failed: {e}, falling back to full image range")
        return [0, 0, W, H]


# ========== Steps 1-5: Generate semantic map and inpaint mask ==========
def build_semantic_change(ctrl: np.ndarray, bbox: list,
                           source_classes: list, target_class: int,
                           bbox_buffer: int = 30,
                           inpaint_buffer: int = 15) -> tuple:
    _, H, W = ctrl.shape
    x1, y1, x2, y2 = bbox  # col_min, row_min, col_max, row_max

    # Step2: bbox加buffer
    x1b = max(0, x1 - bbox_buffer)
    y1b = max(0, y1 - bbox_buffer)
    x2b = min(W, x2 + bbox_buffer)
    y2b = min(H, y2 + bbox_buffer)
    print(f"[MASK] Step2 buffer: col[{x1b}:{x2b}] row[{y1b}:{y2b}]")

    # Step1+3: Find source class pixels within the buffered bbox and perform connected component analysis
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[y1b:y2b, x1b:x2b] = True

    # Find source class pixels
    src_mask = np.zeros((H, W), dtype=bool)
    for c in source_classes:
        src_mask |= (ctrl[0] == c)
    candidate = src_mask & region_mask

    print(f"[MASK] Step1 Candidate pixels: {candidate.sum()}")

    if candidate.sum() == 0:
        print("[MASK] Warning: Source class not found in bbox, expanding to full image search")
        candidate = src_mask
    if candidate.sum() == 0:
        print("[MASK] Error: Source class not found in entire image")
        return ctrl.copy(), np.zeros((H, W), dtype=np.float32), False

    # Connected component analysis: Find the largest connected component 
    labeled, n_comp = connexLabel(candidate)
    if n_comp == 0:
        print("[MASK] Error: Connected component analysis failed")
        return ctrl.copy(), np.zeros((H, W), dtype=np.float32), False

    # Find the largest connected component (or alternatively, the one closest to the bbox center)
    comp_sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
    largest_comp = np.argmax(comp_sizes) + 1

    # Find the connected component closest to the bbox center (aligns better with user intent)
    cx_bbox = (y1 + y2) // 2  # row center
    cy_bbox = (x1 + x2) // 2  # col center
    best_comp = largest_comp
    best_dist = float('inf')
    for i in range(1, n_comp + 1):
        comp_mask = labeled == i
        rows, cols = np.where(comp_mask)
        if len(rows) == 0:
            continue
        cr, cc = rows.mean(), cols.mean()
        dist = ((cr - cx_bbox)**2 + (cc - cy_bbox)**2)**0.5
        # Consider both distance and size (close enough and large enough)
        size = comp_mask.sum()
        score = dist / (size**0.3 + 1)
        if score < best_dist:
            best_dist = score
            best_comp = i

    final_mask = (labeled == best_comp)
    print(f"[MASK] Step3 Connected Components: {n_comp} total, selected #{best_comp}, size={final_mask.sum()}px")

    # Step4: remap semantic labels
    ctrl1 = ctrl.copy()
    ctrl1[0, final_mask] = target_class
    changed = (ctrl1[0] != ctrl[0]).sum()
    print(f"[MASK] Step4 Modified pixels: {changed} ({changed/(H*W)*100:.2f}%)")


    # Step5: Apply morphological dilation to final_mask → inpaint mask
    # Circular structuring element
    y_s, x_s = np.ogrid[-inpaint_buffer:inpaint_buffer+1, -inpaint_buffer:inpaint_buffer+1]
    struct = (x_s**2 + y_s**2) <= inpaint_buffer**2
    inpaint_mask = binary_dilation(final_mask, structure=struct).astype(np.float32)
    # Gaussian blur to soften edges (for more natural boundaries)
    import cv2
    inpaint_mask = cv2.GaussianBlur(inpaint_mask, (31, 31), 10)
    inpaint_mask = (inpaint_mask > 0.15).astype(np.float32)
    print(f"[MASK] Step5 inpaint mask size: {inpaint_mask.sum():.0f}px")

    return ctrl1, inpaint_mask, True


# ========== visualization==========
def label_to_color(arr2d):
    out = np.zeros((*arr2d.shape, 3), dtype=np.uint8)
    for cls, rgb in LUT_COLORS.items():
        out[arr2d == cls] = rgb
    return out

def save_preview(I1_np, I2_np, ctrl, ctrl1, inpaint_mask, bbox, save_dir, image_id):
    # Output 5 separate files in a subfolder named after image_id
    out_dir = os.path.join(save_dir, image_id)
    os.makedirs(out_dir, exist_ok=True)

    I1_vis = (I1_np * 255).clip(0,255).astype(np.uint8)
    I2_vis = (I2_np * 255).clip(0,255).astype(np.uint8) if I2_np.max() <= 1.0 else I2_np.astype(np.uint8)
    M1_vis = label_to_color(ctrl[0])
    M2_vis = label_to_color(ctrl1[0])

    I1_ann = I1_vis.copy()
    I1_ann[inpaint_mask > 0] = (I1_ann[inpaint_mask > 0] * 0.5 + np.array([255,200,0]) * 0.5).astype(np.uint8)
    if bbox:
        x1, y1, x2, y2 = bbox
        I1_ann[y1:y2, max(0,x1-2):x1+2] = [0,255,0]
        I1_ann[y1:y2, x2-2:min(I1_ann.shape[1],x2+2)] = [0,255,0]
        I1_ann[max(0,y1-2):y1+2, x1:x2] = [0,255,0]
        I1_ann[y2-2:min(I1_ann.shape[0],y2+2), x1:x2] = [0,255,0]

    Image.fromarray(I1_vis).save(os.path.join(out_dir, "I1.png"))
    Image.fromarray(M1_vis).save(os.path.join(out_dir, "M1.png"))
    Image.fromarray(I2_vis).save(os.path.join(out_dir, "I2.png"))
    Image.fromarray(M2_vis).save(os.path.join(out_dir, "M2.png"))
    Image.fromarray(I1_ann).save(os.path.join(out_dir, "I1_annotated.png"))

    # change mask
    change_mask = (ctrl1[0] != ctrl[0]).astype(np.uint8) * 255
    Image.fromarray(change_mask).save(os.path.join(out_dir, "change_mask.png"))

    print(f"[VIZ] output directory: {out_dir}")
    print(f"      I1.png / M1.png / I2.png / M2.png / I1_annotated.png / change_mask.png")
    return out_dir

# ==========main function ==========
def run(args):
    image_id = os.path.basename(args.image_path).replace(".tif", "")

    # ---------- read original image and label ----------
    with rasterio.open(args.image_path) as src:
        I1_np     = src.read()[:3].astype(np.float32) / 255.0  # (3,H,W)
        transform = src.transform
        crs       = src.crs
    ctrl = rasterio.open(args.label_path).read().astype(np.uint8)  # (1,H,W)
    _, H, W = ctrl.shape

    # ---------- Step0-A: Text model parses categories----------
    class_info = parse_classes(args.instruction, args.api_key)
    source_classes = class_info["source_classes"]
    target_class   = class_info["target_class"]
    #missing class: auto-select via probability when target_class is null
    if target_class is None:
        target_class = pick_target_class_by_prob(ctrl[0], source_classes)
        print(f"[INFO] 消失类指令，自动选目标类别: {CLASS_NAMES.get(target_class)}")
        class_info["target_desc"] = CLASS_NAMES.get(target_class, str(target_class))
    source_desc    = class_info["source_desc"]
    target_desc    = class_info["target_desc"]
    print(f"\n[INFO] {source_desc} → {target_desc}")
    print(f"[INFO] 源类别ID: {source_classes}, 目标类别ID: {target_class}\n")

    # ---------- Step0-B: VL model loates ----------
    bbox = locate_region_with_vl(
        args.image_path, args.instruction, source_desc, args.api_key, H, W
    )

    # ---------- Steps 1-5: genarate inpaint mask ----------
    ctrl1, inpaint_mask, success = build_semantic_change(
        ctrl, bbox, source_classes, target_class,
        bbox_buffer=30, inpaint_buffer=15
    )

    if not success:
        print("[ERROR] Failed to generate valid change region, exiting")
        return {"source": "", "target": "", "target_class": None, "change_ratio": 0, "success": False}

    # ---------- prompt ----------
    prompt_classes = flair.getMaskedObjects(ctrl1, inpaint_mask[None,:,:].astype(int))
    try:
        if dfPrompts_ext is not None:
            dfPrompts = dfPrompts_ext
        else:
            dfPrompts = pd.read_csv(args.prompts_path, index_col=0)["prompt"].to_dict()
        base_prompt = dfPrompts.get(image_id + ".tif", "")
    except Exception:
        base_prompt = ""
    prompt = flair.PROMPTS_START[0] + prompt_classes + " " + base_prompt + ", high resolution, highly detailed"
    print(f"[PROMPT] {prompt[:150]}...")

    #  to tensor
    I1_tensor    = torch.Tensor(I1_np)                                          # (3,H,W)
    mask_tensor  = torch.Tensor(inpaint_mask)                                   # (H,W)
    ctrl1_color  = np.moveaxis(convert_to_color(ctrl1[0]), 2, 0) / 255.0
    ctrl1_tensor = torch.Tensor(ctrl1_color)                                    # (3,H,W)

    # Load model
    if hasattr(args, '_pipe') and args._pipe is not None:
        pipe = args._pipe
        dfPrompts_ext = getattr(args, '_dfPrompts', {})
    else:
        print("\n[MODEL] load HySCDG pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        controlnet = diffusion.loadControlNet(ControlNet_path=args.controlnet_path, device=device)
        pipe = diffusion.loadPipeline(model_path=args.model_path, controlnet=controlnet, device=device)
        print("[MODEL] successful\n")
        dfPrompts_ext = None

    # ---------- Step6: SD+ControlNet  ----------
    print("[GEN] Step6: SD+ControlNet generate I2...")
    result = pipe(
        prompt                        = [prompt],
        num_inference_steps           = args.inference_steps,
        image                         = [I1_tensor],
        mask_image                    = [mask_tensor],
        output_type                   = "np",
        control_image                 = [ctrl1_tensor],
        controlnet_conditioning_scale = args.conditioning_scale
    )
    I2_np = result[0][0]   # (H,W,3) 0-1
    print("[GEN] finish generation")

    # ---------- save results----------
    out_dir = os.path.join(args.save_dir, image_id)
    os.makedirs(out_dir, exist_ok=True)
    flair.saveRaster(
        os.path.join(out_dir, "I1.tif"),
        (I1_np * 255).astype(np.uint8), transform, crs=crs
    )
    flair.saveRaster(
        os.path.join(out_dir, "I2.tif"),
        (np.moveaxis(I2_np,2,0)*255).astype(np.uint8), transform, crs=crs
    )
    flair.saveRaster(
        os.path.join(out_dir, "M1.tif"),
        ctrl.astype(np.uint8), transform, crs=crs
    )
    flair.saveRaster(
        os.path.join(out_dir, "M2.tif"),
        ctrl1.astype(np.uint8), transform, crs=crs
    )
    save_preview(
        np.moveaxis(I1_np, 0, -1), I2_np,
        ctrl, ctrl1, inpaint_mask, bbox,
        args.save_dir, image_id
    )

    change_ratio = (ctrl1[0] != ctrl[0]).sum() / (H*W) * 100
    print(f"\n{'='*50}")
    print(f"Done! Actual change ratio: {change_ratio:.2f}%")
    print(f"results: {args.save_dir}")
    change_ratio = (ctrl1[0] != ctrl[0]).sum() / (H*W) * 100
    return {"source": class_info["source_desc"], "target": class_info["target_desc"],
            "target_class": target_class, "change_ratio": change_ratio, "success": True}


# ========== CLI ==========
if __name__ == "__main__":
    import random, traceback
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    # Single image mode
    parser.add_argument("--image_path",      type=str, default=None)
    parser.add_argument("--label_path",      type=str, default=None)
    parser.add_argument("--instruction",     type=str, default=None)
    # Batch mode
    parser.add_argument("--jsonl_path",      type=str, default=None,
                        help="批量模式：jsonl文件路径，每行{image_path,label_path,instruction}")
    parser.add_argument("--max_samples",     type=int, default=0, help="批量模式最多处理多少张，0=全部")
    # Fully automatic mode
    parser.add_argument("--images_dir",      type=str, default=None,
                        help="全自动模式：图像文件夹，自动生成指令jsonl后批量生成")
    parser.add_argument("--n_per_image",     type=int, default=3,
                        help="全自动模式：每张图生成几条指令")
    # Common parameters
    parser.add_argument("--model_path",      type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--prompts_path",    type=str, default="/root/autodl-tmp/FLAIR_Prompts.csv")
    parser.add_argument("--save_dir",        type=str, default="/root/autodl-tmp/output_text_guided/auto")
    parser.add_argument("--api_key",         type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.images_dir:
        #Fully Automatic Mode: Generate JSONL first, then batch generate images
        import glob, time
        auto_jsonl = os.path.join(args.save_dir, "auto_instructions.jsonl")
        os.makedirs(args.save_dir, exist_ok=True)

        all_images = sorted(glob.glob(
            os.path.join(args.images_dir, "**", "IMG_*.tif"), recursive=True))
            print(f"[AUTO] Folder contains {len(all_images)} images, generating {args.n_per_image} instructions per image")

        # Skip images with existing instructions
        done_images = set()
        if os.path.exists(auto_jsonl):
            with open(auto_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done_images.add(json.loads(line)["image_path"])
            print(f"[AUTO] Already have instructions for {len(done_images)} images, skipping")

        # Select only images without instructions, then limit new generation count by max_samples
        todo_images = [p for p in all_images if p not in done_images]
        if args.max_samples > 0:
            todo_images = todo_images[:args.max_samples]
        print(f"[AUTO] New instructions to be generated: {len(todo_images)} images\n")

        print(f"[AUTO] Step 1: Generating change instructions with VL...")
        with open(auto_jsonl, "a", encoding="utf-8") as out_f:
            for idx, image_path in enumerate(todo_images):
                if image_path in done_images:
                    continue
                label_path = image_path.replace("/aerial/", "/labels/").replace("IMG", "MSK")
                if not os.path.exists(label_path):
                    print(f"[SKIP] label not found: {label_path}")
                    continue
                print(f"[{idx+1}/{len(all_images)}] {os.path.basename(image_path)}")
                instructions = generate_instructions(image_path, label_path, args.api_key, args.n_per_image)
                for inst in instructions:
                    record = {"image_path": image_path, "label_path": label_path, "instruction": inst}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"  → {inst}")
                out_f.flush()
                time.sleep(0.5)
        print(f"[AUTO] instructions generated, saved to: {auto_jsonl}\n")
        args.jsonl_path = auto_jsonl


    if args.jsonl_path:
        # Batch mode
        # read jsonl，image_path
        groups = defaultdict(list)
        with open(args.jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                groups[rec['image_path']].append(rec)

        image_paths = list(groups.keys())
        # max_samples control
        print(f"[BATCH] 共 {len(image_paths)} 张图，每张随机选1条指令")
        print(f"[BATCH] 共 {len(image_paths)} 张图，每张随机选1条指令")

        # Resume from checkpoint
        done_set = set()
        os.makedirs(args.save_dir, exist_ok=True)
        progress_file = os.path.join(args.save_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file) as f:
                done_set = set(json.load(f))
            print(f"[BATCH] 已完成 {len(done_set)} 张，继续...\n")

        # load prompt CSV
        try:
            dfPrompts = pd.read_csv(args.prompts_path, index_col=0)["prompt"].to_dict()
        except:
            dfPrompts = {}

        # Load the model only once
        print("[MODEL] loading HySCDG pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        controlnet = diffusion.loadControlNet(ControlNet_path=args.controlnet_path, device=device)
        pipe = diffusion.loadPipeline(model_path=args.model_path, controlnet=controlnet, device=device)
        print("[MODEL] loading completed\n")

        total_ok, total_fail = 0, 0
        new_count = 0  
        all_meta = []
        all_meta_file = os.path.join(args.save_dir, "all_instructions.json")
        if os.path.exists(all_meta_file):
            with open(all_meta_file) as f:
                all_meta = json.load(f)
        for idx, image_path in enumerate(image_paths):
            image_id = os.path.basename(image_path).replace(".tif", "")
            if image_id in done_set:
                continue
            # max_samples
            if args.max_samples > 0 and new_count >= args.max_samples:
                break

            records     = groups[image_path]
            label_path  = records[0]['label_path']
            instruction = random.choice(records)['instruction']

            print(f"\n[{idx+1}/{len(image_paths)}] {image_id}")
            print(f"  instruction: {instruction}")

            try:
                # Call run() directly, but skip model loading (already loaded externally)
                args.image_path  = image_path
                args.label_path  = label_path
                args.instruction = instruction
                args._pipe       = pipe
                args._dfPrompts  = dfPrompts
                result_info = run(args)
                #If run() returns None or success=False, treat as source category not in image
                if result_info is None:
                    result_info = {"success": False, "change_ratio": 0}
                if result_info.get("success") == False:
                    print(f"[WARN] Source category not in image, retrying with different instruction...")
                    other_records = [r for r in records if r['instruction'] != instruction]
                    retry_ok = False
                    for retry_rec in other_records:
                        args.instruction = retry_rec['instruction']
                        result_info = run(args)
                        if result_info is None:
                            result_info = {"success": False, "change_ratio": 0}
                        if result_info.get("success") != False and result_info.get("change_ratio", 0) <= 50:
                            instruction = retry_rec['instruction']
                            retry_ok = True
                            break
                    if not retry_ok:
                        import shutil
                        bad_dir = os.path.join(args.save_dir, image_id)
                        if os.path.exists(bad_dir):
                            shutil.rmtree(bad_dir)
                            print(f"[CLEAN] deleted: {bad_dir}")
                        total_fail += 1
                        continue
                # Check if the changed area exceeds 50%; if so, switch to the next instruction and retry
                if result_info.get("change_ratio", 0) > 50:
                    print(f"[WARN] Change area {result_info['change_ratio']:.1f}% > 50%，switch to next instruction...")
                    other_records = [r for r in records if r['instruction'] != instruction]
                    retry_ok = False
                    for retry_rec in other_records:
                        args.instruction = retry_rec['instruction']
                        print(f"[RETRY] switch instruction: {args.instruction}")
                        result_info = run(args)
                        if result_info.get("change_ratio", 0) <= 50:
                            instruction = retry_rec['instruction']
                            retry_ok = True
                            break
                    if not retry_ok:
                        print(f"[WARN] All instructions have change areas > 50%, skipping this image")
                        # Delete the generated folder
                        import shutil
                        bad_dir = os.path.join(args.save_dir, image_id)
                        if os.path.exists(bad_dir):
                            shutil.rmtree(bad_dir)
                            print(f"[CLEAN] deleted: {bad_dir}")
                        total_fail += 1
                        continue
                total_ok += 1
                new_count += 1
                done_set.add(image_id)
                # Save meta to a unified JSON file
                meta = {"image_id": image_id, "instruction": instruction,
                        "source": result_info.get("source",""),
                        "target": result_info.get("target","")}
                all_meta.append(meta)
                with open(all_meta_file, 'w') as mf:
                    json.dump(all_meta, mf, ensure_ascii=False, indent=2)
                with open(progress_file, 'w') as f:
                    json.dump(list(done_set), f)
            except Exception as e:
                total_fail += 1
                print(f"  [ERROR] {e}")
                traceback.print_exc()

        print(f"\nfinish！success:{total_ok} failed:{total_fail}")

    else:
        # single mode
        assert args.image_path and args.label_path and args.instruction, \
            "single mode requires --image_path --label_path --instruction"
        run(args)
