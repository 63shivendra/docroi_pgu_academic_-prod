"""
DocROI-Former v3: Unified Single-Image Forensic Diagnostic Engine
================================================================
Supports BOTH:
  1. Strict Academic Evaluation Mode (--academic):
     - Standard neutral threshold tau = 0.50
     - Zero post-processing heuristics
     - Direct raw model probability and mask outputs
  2. Production Enterprise KYC Mode (default):
     - Neural Entity-Gated Evidence Scoring (suppressing margin glare and blank card false alarms)
     - Multi-hotspot discovery (isolating both photo splicing and fine 5-12 px character alterations)
     - High-resolution 4-panel forensic evidence board export
"""

import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure repo root is on sys.path
repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)

from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer
from src.dataset.entity_tokenizer import OtsuEntityTokenizer


def find_image(img_arg):
    candidates = [
        img_arg,
        os.path.join(r"D:\shivendra pratap singh", os.path.basename(img_arg)),
        os.path.join(r"E:\shivendra pratap singh", os.path.basename(img_arg)),
        os.path.join(repo_dir, os.path.basename(img_arg))
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="DocROI-Former v3: Forensic Diagnostic Engine")
    parser.add_argument("--image", type=str, default=r"D:\shivendra pratap singh\FRONT.png", help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_docroi_former_pguoctv3.pth", help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.50, help="Tamper threshold (default: 0.50)")
    parser.add_argument("--academic", action="store_true", help="Run in strict Academic CVPR mode (pure tau=0.50, unmodulated)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    image_path = find_image(args.image)
    if not image_path:
        print(f"Error: Could not locate target image at '{args.image}'.")
        print("Please verify the file path exists.")
        sys.exit(1)

    mode_label = "ACADEMIC CVPR (Strict tau=0.50)" if args.academic else "ENTERPRISE PRODUCTION (Entity-Gated)"
    print("=" * 80)
    print(f"  DocROI-Former v3 Forensic Inspector [{mode_label}]")
    print("=" * 80)
    print(f"  Target Image:     {image_path}")
    print(f"  Model Checkpoint: {args.checkpoint}")
    print(f"  Computing Device: {args.device}")
    print("=" * 80)

    cfg = DocROIConfig()
    model = DocROIFormer(config=cfg).to(args.device)

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_dir, ckpt_path)

    if not os.path.exists(ckpt_path):
        # Fallback to v2 if pguoctv3 not found at that relative path
        alt_path = os.path.join(repo_dir, "checkpoints", "best_docroi_former_v2_pgu.pth")
        if os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            print(f"Error: Checkpoint '{ckpt_path}' not found.")
            sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load and preprocess image
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        print(f"Error: OpenCV could not decode image at '{image_path}'.")
        sys.exit(1)

    orig_h, orig_w = orig_bgr.shape[:2]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Standard model input resolution is 384x384
    input_size = 384
    resized_rgb = cv2.resize(orig_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    norm_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    norm_tensor = ((norm_tensor - mean) / std).unsqueeze(0).to(args.device)

    with torch.no_grad():
        preds = model(norm_tensor)
        mask_logits = preds["mask_logits"]
        global_logits = preds["global_logits"]

        prob_map_384 = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
        global_prob = torch.sigmoid(global_logits)[0, 0].item()

        # Extract Stage-1 Neural Semantic Relevance Map if available
        if "relevance_map" in preds and preds["relevance_map"] is not None:
            rel_384 = F.interpolate(preds["relevance_map"], size=(input_size, input_size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        else:
            rel_384 = np.ones((input_size, input_size), dtype=np.float32)

    # Resize probability and relevance maps to original image resolution
    prob_map_raw = cv2.resize(prob_map_384, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    relevance_orig = cv2.resize(rel_384, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Morphological entity mask as dual verification
    tokenizer = OtsuEntityTokenizer(kernel_size=3, dilation_iterations=2)
    entity_mask_morph = tokenizer.extract_entity_mask(orig_rgb)
    combined_relevance = np.clip(relevance_orig * 0.7 + entity_mask_morph * 0.3, 0.0, 1.0)

    if args.academic:
        # Academic mode: Strict unweighted raw model probability
        prob_map_active = prob_map_raw
    else:
        # Production mode: Neural Entity-Gated Modulation
        # Attenuates inert blank margin noise while preserving high-contrast entities
        prob_map_active = prob_map_raw * combined_relevance

    # Mathematical Evidence Scoring
    entity_evidence = float(np.sum((prob_map_raw > args.threshold) * combined_relevance))
    total_entity_mass = float(np.sum(combined_relevance)) + 1e-6
    entity_tamper_density = (entity_evidence / total_entity_mass) * 100.0

    peak_prob = float(prob_map_active.max() * 100.0)
    bg_noise = float(prob_map_raw[combined_relevance < 0.2].mean() * 100.0) if (combined_relevance < 0.2).any() else 0.0
    fg_mean = float(prob_map_raw[combined_relevance >= 0.2].mean() * 100.0) if (combined_relevance >= 0.2).any() else 0.0

    binary_mask = (prob_map_active > args.threshold).astype(np.uint8)
    flagged_pixels = int(binary_mask.sum())

    # Production decision: Global head OR high-confidence entity-gated evidence
    if args.academic:
        is_tampered = bool(global_prob >= args.threshold)
    else:
        is_tampered = bool((global_prob >= 0.50) or (entity_evidence >= 12.0) or (peak_prob >= 40.0 and flagged_pixels >= 8))

    # Find bounding boxes on tampered regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= 6:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h, area))
    boxes.sort(key=lambda b: b[4], reverse=True)

    # Multi-Hotspot Discovery (Primary & Secondary Peaks)
    hotspots = []
    peak_y, peak_x = np.unravel_index(np.argmax(prob_map_active), prob_map_active.shape)
    hotspots.append({
        "name": "Hotspot #1 (Primary)",
        "x": peak_x, "y": peak_y,
        "score": peak_prob,
        "type": "Primary Tamper Hotspot",
        "color": (255, 0, 0)  # Red
    })

    suppress_r = max(30, int(min(orig_h, orig_w) * 0.08))
    sec_map = prob_map_active.copy()
    y_min, y_max = max(0, peak_y - suppress_r), min(orig_h, peak_y + suppress_r + 1)
    x_min, x_max = max(0, peak_x - suppress_r), min(orig_w, peak_x + suppress_r + 1)
    sec_map[y_min:y_max, x_min:x_max] = 0.0

    if sec_map.max() > 0.15:
        s_peak_y, s_peak_x = np.unravel_index(np.argmax(sec_map), sec_map.shape)
        hotspots.append({
            "name": "Hotspot #2 (Secondary)",
            "x": s_peak_x, "y": s_peak_y,
            "score": float(sec_map.max() * 100.0),
            "type": "Secondary Tamper Hotspot",
            "color": (0, 165, 255)  # Orange / Amber
        })

    print("\n--- FORENSIC INFERENCE METRICS ---")
    print(f"  Target File:                 {os.path.basename(image_path)} ({orig_w}x{orig_h})")
    print(f"  Document-Level Fraud Score:  {global_prob * 100:.2f}% (P_doc Head)")
    print(f"  Entity-Gated Evidence Score: {entity_evidence:.2f} px-mass")
    print(f"  Entity Tamper Density:       {entity_tamper_density:.4f}%")
    print(f"  Peak Localized Forgery Prob: {peak_prob:.2f}%")
    print(f"  Blank Margin Noise Level:    {bg_noise:.4f}%")
    print(f"  Foreground Entity Score:     {fg_mean:.2f}%")
    print(f"  Flagged Pixels (tau={args.threshold:.2f}): {flagged_pixels}")
    print(f"  Detected Tamper Hotspots:    {len(hotspots)}")
    for h in hotspots:
        print(f"    [{h['name']}] Coord [X:{h['x']}, Y:{h['y']}] | Intensity: {h['score']:.2f}%")

    if is_tampered:
        print(f"\n  >>> FORENSIC VERDICT: [TAMPERED / FRAUD DETECTED]")
    else:
        print(f"\n  >>> FORENSIC VERDICT: [GENUINE / CLEAN DOCUMENT]")

    # =========================================================================
    # 4-Panel Forensic Evidence Board Export
    # =========================================================================
    fig = plt.figure(figsize=(20, 11), dpi=200)
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    # Panel 1: Original Image + Bounding Boxes & Hotspots
    ax1 = fig.add_subplot(gs[0, 0])
    annotated_rgb = orig_rgb.copy()
    for h in hotspots:
        hx, hy = h["x"], h["y"]
        col = h["color"]
        cv2.circle(annotated_rgb, (hx, hy), 22, col, 3)
        cv2.putText(annotated_rgb, f"{h['name']}: {h['score']:.1f}%",
                    (max(10, hx - 80), max(25, hy - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    for (bx, by, bw, bh, _) in boxes[:4]:
        cv2.rectangle(annotated_rgb, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2)
    ax1.imshow(annotated_rgb)
    v_color = "red" if is_tampered else "green"
    v_str = "TAMPERED" if is_tampered else "GENUINE"
    ax1.set_title(f"1. Target Document: {os.path.basename(image_path)} | Verdict: {v_str} (P_doc={global_prob*100:.1f}%)",
                  fontsize=12, fontweight="bold", color=v_color)
    ax1.axis("off")

    # Panel 2: Normalized Tamper Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    max_val = max(float(prob_map_active.max()), 1e-6)
    norm_heatmap = prob_map_active / max_val
    im2 = ax2.imshow(norm_heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Relative Forgery Intensity")
    ax2.set_title(f"2. Entity-Gated Forgery Heatmap (Peak: {peak_prob:.1f}%, Margin Noise: {bg_noise:.2f}%)",
                  fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Binary Localization Mask
    ax3 = fig.add_subplot(gs[1, 0])
    gray_bg = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    overlay_rgb = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)
    overlay_rgb[binary_mask > 0] = [255, 40, 40]  # Red highlight
    ax3.imshow(overlay_rgb)
    ax3.set_title(f"3. Binary Tamper Contours (tau={args.threshold:.2f}, Altered Pixels: {flagged_pixels})",
                  fontsize=12, fontweight="bold")
    ax3.axis("off")

    # Panel 4: Microscopic Zoom on Detected Hotspots
    ax4 = fig.add_subplot(gs[1, 1])
    crop_r = 65
    crops = []
    for h in hotspots[:2]:
        hx, hy = h["x"], h["y"]
        cy1, cy2 = max(0, hy - crop_r), min(orig_h, hy + crop_r)
        cx1, cx2 = max(0, hx - crop_r), min(orig_w, hx + crop_r)
        c_rgb = orig_rgb[cy1:cy2, cx1:cx2]
        c_heat = norm_heatmap[cy1:cy2, cx1:cx2]
        c_col = cv2.applyColorMap((c_heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        c_col = cv2.cvtColor(c_col, cv2.COLOR_BGR2RGB)
        c_blend = cv2.addWeighted(c_rgb, 0.45, c_col, 0.55, 0)
        c_blend = cv2.resize(c_blend, (180, 180))
        cv2.putText(c_blend, f"{h['name'][:12]}: {h['score']:.1f}%", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        crops.append(c_blend)

    if len(crops) == 2:
        combined_zoom = np.hstack([crops[0], np.ones((180, 12, 3), dtype=np.uint8) * 255, crops[1]])
        ax4.imshow(combined_zoom)
        ax4.set_title("4. Microscopic Hotspot Zoom [Left: Primary Peak | Right: Secondary Hotspot]",
                      fontsize=12, fontweight="bold")
    elif len(crops) == 1:
        ax4.imshow(crops[0])
        ax4.set_title(f"4. Microscopic Hotspot Zoom (Peak Score: {hotspots[0]['score']:.1f}%)",
                      fontsize=12, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No Tamper Hotspots Detected Above Threshold",
                 ha="center", va="center", fontsize=14, color="gray")
    ax4.axis("off")

    img_stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(repo_dir, "reports", "pgumodelresutls")
    os.makedirs(out_dir, exist_ok=True)
    out_board_path = os.path.join(out_dir, f"audit_{img_stem}_result.png")
    plt.savefig(out_board_path, bbox_inches="tight")
    plt.close()

    print(f"\n[Audit Board Saved] Full 4-Panel Evidence Board exported to:\n  -> {out_board_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
