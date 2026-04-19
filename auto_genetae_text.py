"""
Step1: Automatically generate variation instructions for the image folder:
input：label tif→ Qwen-VL
output：instructions.jsonl，
{"image_path": "...", "label_path": "...", "instruction": "..."}

Usage：
  python /root/autodl-tmp/auto_genetae_text.py \
    --images_dir "/root/autodl-tmp/flair_aerial_train/aerial/Z18_UU" \
    --save_path  "/root/autodl-tmp/auto_instructions.jsonl" \
    --api_key    "……" \
    --n_per_image 3 \
    --max_images  5
"""

import os, json, argparse, base64, io, time, glob
import numpy as np
import tifffile
from PIL import Image
from openai import OpenAI

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VL_MODEL = "qwen-vl-plus"

# FLAIR 19-class color mapping
FLAIR_COLORS = {
    1:'#db0e9a', 2:'#938e7b', 3:'#f80c00', 4:'#a97101',
    5:'#1553ae', 6:'#194a26', 7:'#46e483', 8:'#f3a60d',
    9:'#660082', 10:'#55ff00', 11:'#fff30d', 12:'#e4df7c',
    13:'#3de6eb', 14:'#ffffff', 15:'#8ab3a0', 16:'#6b714f',
    17:'#c5dc42', 18:'#9999ff', 19:'#000000',
}

CLASS_NAMES = {
    1:"building",           2:"pervious surface",    3:"impervious surface",
    4:"bare soil",          5:"water",               6:"coniferous",
    7:"deciduous",          8:"brushwood",           9:"vineyard",
    10:"herbaceous vegetation", 11:"agricultural land", 12:"plowed land",
    13:"swimming pool",     14:"snow",               15:"clear cut",
    16:"mixed",             17:"ligneous",           18:"greenhouse",
    19:"other"
}

# Color legend for VL models (Color → English class name, with aliases in parentheses for clarity).）
COLOR_LEGEND = (
    "Color legend (color in image → land cover class):\n"
    "pink        = building (house, residential, rooftop)\n"
    "gray        = pervious surface (permeable ground, gravel, unpaved)\n"
    "red         = impervious surface (road, pavement, asphalt, street)\n"
    "brown       = bare soil (bare land, dirt, empty ground)\n"
    "dark blue   = water (river, lake, pond, stream)\n"
    "dark green  = coniferous (pine forest, evergreen trees)\n"
    "bright green= deciduous (forest, trees, woodland)\n"
    "orange      = brushwood (shrubs, bushes, scrubland)\n"
    "purple      = vineyard (grape field)\n"
    "yellow-green= herbaceous vegetation (grass, meadow, lawn)\n"
    "yellow      = agricultural land (farmland, crops, field)\n"
    "light yellow= plowed land (tilled field, cultivated land)\n"
    "cyan        = swimming pool (pool, water feature)\n"
    "white       = snow (snow cover)\n"
    "black       = other\n"
)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def label_to_base64(label_path: str) -> str:
    """tif label(0-19) → FLAIR image → base64"""
    arr = tifffile.imread(label_path)  # (H,W) 0-19
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for val, hex_color in FLAIR_COLORS.items():
        rgb[arr == val] = hex_to_rgb(hex_color)
    img_pil = Image.fromarray(rgb).resize((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_class_stats(label_path: str) -> str:
    """Calculate the percentage and rough position of each class, acting as auxiliary textual context"""
    arr = tifffile.imread(label_path)
    H, W = arr.shape
    stats = []
    for c in range(1, 20):
        ratio = (arr == c).sum() / (H * W)
        if ratio < 0.03:
            continue
        rows, cols = np.where(arr == c)
        cr, cc = rows.mean() / H, cols.mean() / W
        row_pos = "Top" if cr < 0.4 else ("Bottom" if cr > 0.6 else "Middle")
        col_pos = "Left" if cc < 0.4 else ("Right" if cc > 0.6 else "Middle")
        if row_pos == "Middle" and col_pos == "Middle":
            pos = "Center"
        elif row_pos == "Middle":
            pos = col_pos
        elif col_pos == "Middle":
            pos = row_pos
        else:
            pos = row_pos + col_pos 
            
        stats.append(f"{CLASS_NAMES[c]}({ratio*100:.0f}%,{pos})")
    return ", ".join(stats) if stats else "Various features"


def img_to_base64(image_path: str) -> str:
    """original tif → base64"""
    import rasterio
    with rasterio.open(image_path) as src:
        img_np = src.read()[:3]
    img_pil = Image.fromarray(np.moveaxis(img_np, 0, -1).astype(np.uint8))
    img_pil = img_pil.resize((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_instructions(image_path: str, label_path: str, api_key: str, n: int) -> list:
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    # Pass both the original image and the colored label map
    b64_img = img_to_base64(image_path)
    b64_lbl = label_to_base64(label_path)
    stats_str = get_class_stats(label_path)

    prompt = f"""{COLOR_LEGEND}

The first image is the original aerial remote sensing image, the second image is the corresponding semantic segmentation label visualization.
Main land cover statistics: {stats_str}.

As a remote sensing change detection dataset construction expert, generate {n} realistic and reasonable land cover change instructions based on the land cover distribution in the images.

Requirements:
1. Each instruction must clearly describe one geographically reasonable change (e.g., "buildings on the right change to trees", "river in the middle disappears", "buildings on top change to bare soil")
2. Must include a clear spatial position (top-left / bottom-right / center area / leftmost / upper area, etc.)
3. The target class must follow common geographic sense and differ from the source class (e.g., isolated buildings cannot appear inside forest, trees cannot appear in rivers, trees cannot grow in the middle of roads)
4. The source class must actually exist in the image (refer to the statistics above)
5. The source class must NOT be a dominant large region to avoid changing the entire image (e.g., do not change a large connected forest block to grassland, or a large central bare soil region to vegetation)
6. Use natural language like a real user (English)
7. Both replacement type ("change X to Y", "replace X with Y") and disappearance type ("remove X", "X disappears") are allowed, ratio approximately 7:3
8. Keep each instruction under 15 words

Output ONLY a JSON array, no other text:
["instruction1", "instruction2", "instruction3"]"""

    # Count the categories actually present in the image (occupancy > 1%)
    arr = tifffile.imread(label_path)
    H, W = arr.shape
    present_classes = set(c for c in range(1,20) if (arr==c).sum()/(H*W) > 0.01)

    try:
        resp = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_lbl}"}},
                {"type": "text", "text": prompt}
            ]}],
            temperature=0.8
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        instructions = json.loads(raw)
        assert isinstance(instructions, list) and len(instructions) > 0
        instructions = [str(i) for i in instructions[:n]]
        return instructions
    except Exception as e:
        print(f"  [WARN] VL generation failed: {e}，using fallback")
        return _fallback(n)


def _fallback(n: int) -> list:
    templates = [
        "Change the forest in the upper left corner into a building",
        "Replace the vegetation in the lower right corner with a road",
        "Change the farmland in the center area into bare soil",
        "Remove the water body on the left side",
        "Change the grassland on the top into a building",
    ]
    return templates[:n]


def run(args):
    all_images = sorted(glob.glob(
        os.path.join(args.images_dir, "**", "IMG_*.tif"), recursive=True))
    if args.max_images > 0:
        all_images = all_images[:args.max_images]
    print(f"[INFO] Total {len(all_images)} images, generating {args.n_per_image} instructions per image")
    print(f"[INFO] Expected to generate {len(all_images) * args.n_per_image} instructions in total\n")

    # Resume from checkpoint
    done_images = set()
    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            for line in f:
                try:
                    done_images.add(json.loads(line)["image_path"])
                except:
                    pass
        print(f"[INFO] Records exist, skipping {len(done_images)} images\n")

    total = 0
    with open(args.save_path, "a", encoding="utf-8") as out_f:
        for idx, image_path in enumerate(all_images):
            if image_path in done_images:
                continue

            label_path = image_path.replace("/aerial/", "/labels/").replace("IMG", "MSK")
            if not os.path.exists(label_path):
                print(f"[SKIP] Label not found: {label_path}")
                continue

            print(f"[{idx+1}/{len(all_images)}] {os.path.basename(image_path)}")
            instructions = generate_instructions(image_path, label_path, args.api_key, args.n_per_image)

            for inst in instructions:
                record = {
                    "image_path": image_path,
                    "label_path": label_path,
                    "instruction": inst
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  → {inst}")
                total += 1

            out_f.flush()
            time.sleep(0.5)

    print(f"\nDone! Generated {total} instructions in total")
    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir",  type=str, required=True)
    p.add_argument("--save_path",   type=str, default="/root/autodl-tmp/auto_instructions.jsonl")
    p.add_argument("--api_key",     type=str, required=True)
    p.add_argument("--n_per_image", type=int, default=3)
    p.add_argument("--max_images",  type=int, default=0, help="0=all")
    args = p.parse_args()
    run(args)