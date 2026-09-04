"""
Particle Image Reveal Animation — v2 (neon edge-glow style)
-------------------------------------------------------------
Upgrade of the original filled-color-particle version, retuned to match a
reference clip where the reveal reads as a glowing NEON LINE-ART drawing
(bright cyan/magenta/gold/white outlines with dim colored fill underneath)
plus a scattered glitter-dust halo, rather than a full-color pixel mosaic.

What changed vs the original script, and why:

1. EDGE + FILL split. Every foreground pixel is now classified as an
   "edge" pixel (from a Canny edge map of the source) or a "fill" pixel
   (everything else). Edge pixels are sampled at near-full density, made
   bigger, and pushed hard toward saturated/near-white neon. Fill pixels
   are sampled much more sparsely and kept dimmer/smaller. That's what
   makes outlines pop and the interior read as soft glow instead of a
   flat color mosaic.

2. Sparkle-dust layer. A separate, smaller set of particles is scattered
   in a jittered halo around the edge pixels (not tied to the source
   image's colors) using a fixed neon palette. These don't fly in from
   across the canvas — they fade/twinkle in roughly in place, which is
   what reads as "glitter" around the linework in the reference.

3. Dual-pass bloom. The old single gaussian bloom pass is replaced with
   two passes: a small-sigma/high-cutoff "tight" pass for a crisp glowing
   core, and a large-sigma/lower-cutoff "soft" pass for the ambient halo
   that bleeds into the background. Combining both (instead of one blur)
   is what gives a real neon-tube look instead of a uniform haze.

4. Per-particle size/alpha/twinkle arrays instead of one global brightness
   -> size mapping, so edge/fill/sparkle groups can each look different
   without three separate render passes.

5. Fixed a bug from the original file: MIN_COVERAGE and PARTICLE_CEILING
   were each defined twice (a leftover from iterating on values) — Python
   silently used the second definition, so the first was dead code. Only
   one definition of each now.

Requirements: numpy, matplotlib, Pillow, scipy, opencv-python (cv2), and
ffmpeg on PATH. scikit-image is optional (used for Otsu auto-threshold,
same as the original).

You need to supply your own source image (PNG/JPG) — this script does not
include or generate any copyrighted artwork itself.
"""

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation
import subprocess
import os
import shutil
import colorsys

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from skimage.filters import threshold_otsu
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# ----------------------- CONFIG -----------------------
SOURCE_IMAGE   = "source.png"      # your input image
OUTPUT_VIDEO   = "particle_reveal.mp4"
FRAME_DIR      = "frames_tmp"

CANVAS_SIZE    = (1080, 1920)      # (width, height) — vertical, reel-style

FPS            = 30
FLY_IN_SECONDS = 5.5
HOLD_SECONDS   = 2.5
STAGGER        = 0.6               # 0 = all particles move together, 1 = max stagger

AUTO_THRESHOLD = True
BRIGHTNESS_THRESHOLD = 25          # only used if AUTO_THRESHOLD = False
THRESHOLD_MARGIN = 10

# --- Edge particles (the neon outlines — these carry the "reveal") ---
EDGE_LOW, EDGE_HIGH = 60, 160       # Canny thresholds (raise both if you get
                                     # too much noisy detail from JPEG artifacts;
                                     # lower both if fine linework is being missed)
EDGE_DILATE_PX      = 1             # thickens the edge mask by N px so it isn't
                                     # a hairline that disappears when downsampled
EDGE_KEEP_FRACTION  = 0.95          # fraction of edge pixels to keep as particles
EDGE_CEILING        = 45000
EDGE_SIZE           = 6.5
EDGE_SATURATION_BOOST = 2.3
EDGE_BRIGHTNESS_BOOST = 1.6
EDGE_MAX_ALPHA      = 1.0

# --- Fill particles (dim color underneath the outlines) ---
FILL_COVERAGE       = 0.05          # fraction of interior (non-edge) foreground
                                     # pixels sampled — deliberately sparse
FILL_CEILING        = 20000
FILL_SIZE           = 3.5
FILL_SATURATION_BOOST = 1.3
FILL_BRIGHTNESS_BOOST = 1.0
FILL_MAX_ALPHA      = 0.65          # capped dimmer than edges on purpose

# --- Sparkle dust (ambient glitter scattered around the linework) ---
SPARKLE_COUNT       = 2600
SPARKLE_SPREAD_PX   = 55            # how far sparkles jitter from an edge pixel
SPARKLE_SIZE_RANGE  = (1.5, 5.0)
SPARKLE_MAX_ALPHA   = 0.9
SPARKLE_PALETTE     = [             # fixed neon palette, not sampled from image
    (1.00, 1.00, 1.00),   # white
    (0.35, 1.00, 1.00),   # cyan
    (1.00, 0.30, 0.95),   # magenta
    (1.00, 0.85, 0.30),   # gold
    (0.55, 0.55, 1.00),   # violet
]

# --- Shared look ---
HALO_LAYER          = True          # soft under-glow disc drawn behind each particle
HALO_SIZE_MULT      = 4.0
EDGE_HALO_ALPHA     = 0.16
FILL_HALO_ALPHA     = 0.0           # fill particles skip the halo (keeps interior
                                     # from turning muddy/gray)
SPARKLE_HALO_ALPHA  = 0.35

TWINKLE             = True
EDGE_TWINKLE_AMP    = 0.12
FILL_TWINKLE_AMP    = 0.12
SPARKLE_TWINKLE_AMP = 0.45          # sparkles flicker much more — that's the glitter

# --- Dual-pass bloom (replaces the old single-pass glow) ---
GLOW                = True
TIGHT_GLOW_SIGMA    = 2.5
TIGHT_GLOW_CUTOFF   = 150
SOFT_GLOW_SIGMA     = 14
SOFT_GLOW_CUTOFF    = 90
SOFT_GLOW_WEIGHT    = 0.55          # soft pass contributes less than tight pass,
                                     # so backgrounds don't wash out gray

BG_COLOR       = "black"
SCATTER_MODE   = "random"          # "random" or "burst" — only affects edge/fill
                                     # particles; sparkles fade in place
RANDOM_SEED    = 7
# --------------------------------------------------------


def enhance_colors(colors, saturation_boost, brightness_boost):
    """Boost saturation/brightness in HSV space so colors read as vivid neon."""
    if len(colors) == 0:
        return colors
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in colors])
    hsv[:, 1] = np.clip(hsv[:, 1] * saturation_boost, 0, 1)
    hsv[:, 2] = np.clip(hsv[:, 2] * brightness_boost, 0, 1)
    rgb = np.array([colorsys.hsv_to_rgb(*c) for c in hsv])
    return rgb


def auto_detect_threshold(brightness, margin):
    """Find the real background/subject split point for this image instead of
    assuming background is pure black. Screenshots and re-compressed images
    often have background around 25-30, not 0."""
    if _HAS_SKIMAGE:
        base = threshold_otsu(brightness)
    else:
        base = np.percentile(brightness, 40)
    return base + margin


def compute_edge_mask(gray_uint8, low, high, dilate_px):
    """Canny edge map of the (already-resized) foreground image. Falls back to
    a simple gradient-magnitude threshold if opencv isn't available."""
    if _HAS_CV2:
        blurred = cv2.GaussianBlur(gray_uint8, (3, 3), 0)
        edges = cv2.Canny(blurred, low, high) > 0
    else:
        gy, gx = np.gradient(gray_uint8.astype(np.float32))
        mag = np.hypot(gx, gy)
        edges = mag > np.percentile(mag, 90)
    if dilate_px > 0:
        struct = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
        edges = binary_dilation(edges, structure=struct)
    return edges


def load_particle_sets(image_path, canvas_w, canvas_h):
    """Load image, split foreground into edge/fill pixel sets, sample each at
    its own density, and return one combined particle array with per-particle
    size/alpha/twinkle attributes so the render loop can stay generic."""
    img = Image.open(image_path).convert("RGB")

    img_ratio = img.width / img.height
    canvas_ratio = canvas_w / canvas_h
    if img_ratio > canvas_ratio:
        new_w = int(canvas_w * 0.85)
        new_h = int(new_w / img_ratio)
    else:
        new_h = int(canvas_h * 0.85)
        new_w = int(new_h * img_ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    arr = np.array(img)
    brightness = arr.mean(axis=2)

    if AUTO_THRESHOLD:
        thresh = auto_detect_threshold(brightness, THRESHOLD_MARGIN)
        print(f"Auto-detected background threshold: {thresh:.1f}")
    else:
        thresh = BRIGHTNESS_THRESHOLD

    fg_mask = brightness > thresh
    fg_pct = 100 * fg_mask.sum() / fg_mask.size
    print(f"Foreground pixels: {fg_pct:.1f}% of image "
          f"({'looks reasonable' if fg_pct < 60 else 'still high — try raising THRESHOLD_MARGIN'})")

    edge_mask = compute_edge_mask(brightness.astype(np.uint8), EDGE_LOW, EDGE_HIGH,
                                   EDGE_DILATE_PX) & fg_mask
    fill_mask = fg_mask & ~edge_mask

    def offset_positions(xs, ys):
        offset_x = (canvas_w - new_w) / 2
        offset_y = (canvas_h - new_h) / 2
        fx = xs + offset_x
        fy = canvas_h - (ys + offset_y)
        return np.stack([fx, fy], axis=1)

    # ---- edge particles ----
    eys, exs = np.where(edge_mask)
    n_edge = min(int(len(exs) * EDGE_KEEP_FRACTION), EDGE_CEILING, len(exs))
    if len(exs) > n_edge:
        idx = np.random.choice(len(exs), n_edge, replace=False)
        exs, eys = exs[idx], eys[idx]
    edge_colors = enhance_colors(arr[eys, exs] / 255.0, EDGE_SATURATION_BOOST,
                                  EDGE_BRIGHTNESS_BOOST)
    edge_pos = offset_positions(exs, eys)
    print(f"Edge particles: {len(edge_pos)}")

    # ---- fill particles ----
    fys, fxs = np.where(fill_mask)
    n_fill = min(max(1, int(len(fxs) * FILL_COVERAGE)), FILL_CEILING, len(fxs))
    if len(fxs) > n_fill:
        idx = np.random.choice(len(fxs), n_fill, replace=False)
        fxs, fys = fxs[idx], fys[idx]
    fill_colors = enhance_colors(arr[fys, fxs] / 255.0, FILL_SATURATION_BOOST,
                                  FILL_BRIGHTNESS_BOOST)
    fill_pos = offset_positions(fxs, fys)
    print(f"Fill particles: {len(fill_pos)}")

    # ---- sparkle dust (jittered around edge positions, fixed palette) ----
    n_sparkle = min(SPARKLE_COUNT, max(1, len(edge_pos) * 4))
    src_idx = np.random.choice(len(edge_pos), n_sparkle, replace=True)
    jitter = np.random.normal(0, SPARKLE_SPREAD_PX, size=(n_sparkle, 2))
    sparkle_pos = edge_pos[src_idx] + jitter
    sparkle_pos[:, 0] = np.clip(sparkle_pos[:, 0], 0, canvas_w)
    sparkle_pos[:, 1] = np.clip(sparkle_pos[:, 1], 0, canvas_h)
    palette = np.array(SPARKLE_PALETTE)
    sparkle_colors = palette[np.random.randint(0, len(palette), n_sparkle)]
    print(f"Sparkle particles: {len(sparkle_pos)}")

    n_e, n_f, n_s = len(edge_pos), len(fill_pos), len(sparkle_pos)
    positions = np.concatenate([edge_pos, fill_pos, sparkle_pos], axis=0)
    colors = np.concatenate([edge_colors, fill_colors, sparkle_colors], axis=0)

    sizes = np.concatenate([
        np.full(n_e, EDGE_SIZE),
        np.full(n_f, FILL_SIZE),
        np.random.uniform(*SPARKLE_SIZE_RANGE, n_s),
    ])
    max_alpha = np.concatenate([
        np.full(n_e, EDGE_MAX_ALPHA),
        np.full(n_f, FILL_MAX_ALPHA),
        np.full(n_s, SPARKLE_MAX_ALPHA),
    ])
    halo_alpha = np.concatenate([
        np.full(n_e, EDGE_HALO_ALPHA),
        np.full(n_f, FILL_HALO_ALPHA),
        np.full(n_s, SPARKLE_HALO_ALPHA),
    ])
    twinkle_amp = np.concatenate([
        np.full(n_e, EDGE_TWINKLE_AMP),
        np.full(n_f, FILL_TWINKLE_AMP),
        np.full(n_s, SPARKLE_TWINKLE_AMP),
    ])
    # sparkles fade/twinkle roughly in place instead of flying across the canvas
    is_stationary = np.concatenate([
        np.zeros(n_e, dtype=bool),
        np.zeros(n_f, dtype=bool),
        np.ones(n_s, dtype=bool),
    ])

    return positions, colors, sizes, max_alpha, halo_alpha, twinkle_amp, is_stationary


def make_start_positions(n, canvas_w, canvas_h, mode="random"):
    if mode == "burst":
        cx, cy = canvas_w / 2, canvas_h / 2
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.uniform(canvas_w * 0.6, canvas_w * 1.2, n)
        sx = cx + np.cos(angles) * radii
        sy = cy + np.sin(angles) * radii
    else:
        sx = np.random.uniform(0, canvas_w, n)
        sy = np.random.uniform(0, canvas_h, n)
    return np.stack([sx, sy], axis=1)


def ease_out_cubic(t):
    t = np.clip(t, 0, 1)
    return 1 - (1 - t) ** 3


def render_frames(positions, colors, sizes, max_alpha, halo_alpha, twinkle_amp,
                   is_stationary, canvas_w, canvas_h, fly_frames, hold_frames,
                   stagger, halo_layer, halo_size_mult, glow, bg_color, seed):
    np.random.seed(seed)
    n = len(positions)

    start_positions = make_start_positions(n, canvas_w, canvas_h, SCATTER_MODE)
    start_positions[is_stationary] = positions[is_stationary]

    stagger_offsets = np.random.uniform(0, stagger, n)
    particle_durations = 1 - stagger_offsets

    twinkle_phase = np.random.uniform(0, 2 * np.pi, n)
    twinkle_speed = np.random.uniform(0.1, 0.25, n)

    os.makedirs(FRAME_DIR, exist_ok=True)
    dpi = 100
    fig_w, fig_h = canvas_w / dpi, canvas_h / dpi
    total_frames = fly_frames + hold_frames

    for f in range(total_frames):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.set_xlim(0, canvas_w)
        ax.set_ylim(0, canvas_h)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if f < fly_frames:
            global_t = f / max(fly_frames - 1, 1)
            local_t = (global_t - stagger_offsets) / np.clip(particle_durations, 1e-6, None)
            eased_t = ease_out_cubic(local_t)
            current_pos = start_positions + (positions - start_positions) * eased_t[:, None]
            base_alpha = np.clip(local_t.flatten() * 2, 0, 1)
        else:
            current_pos = positions
            base_alpha = np.ones(n)
            if TWINKLE:
                hold_f = f - fly_frames
                flicker = 1.0 + twinkle_amp * np.sin(hold_f * twinkle_speed + twinkle_phase)
                base_alpha = base_alpha * flicker

        alpha = np.clip(base_alpha, 0, 1) * max_alpha

        if halo_layer:
            ax.scatter(current_pos[:, 0], current_pos[:, 1],
                       c=colors, s=sizes * halo_size_mult,
                       alpha=np.clip(alpha * (halo_alpha / np.maximum(max_alpha, 1e-6)), 0, 1),
                       edgecolors="none")

        ax.scatter(current_pos[:, 0], current_pos[:, 1],
                   c=colors, s=sizes, alpha=np.clip(alpha, 0, 1),
                   edgecolors="none")

        fig.canvas.draw()
        frame_path = os.path.join(FRAME_DIR, f"frame_{f:05d}.png")
        fig.savefig(frame_path, facecolor=bg_color)
        plt.close(fig)

        if glow:
            apply_dual_glow(frame_path)

    return total_frames


def apply_dual_glow(frame_path):
    """Two-pass bloom: a tight, high-cutoff pass for a crisp glowing core, and
    a soft, lower-cutoff pass (down-weighted) for the ambient halo. Screen-
    blending both onto the original is what reads as neon-tube glow instead
    of a uniform haze."""
    img = Image.open(frame_path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    brightness = arr.mean(axis=2, keepdims=True)

    def highlight_pass(cutoff, sigma, weight):
        mask = np.clip((brightness - cutoff) / (255 - cutoff), 0, 1)
        highlighted = arr * mask
        blurred = gaussian_filter(highlighted, sigma=(sigma, sigma, 0))
        return blurred * weight

    tight = highlight_pass(TIGHT_GLOW_CUTOFF, TIGHT_GLOW_SIGMA, 1.0)
    soft = highlight_pass(SOFT_GLOW_CUTOFF, SOFT_GLOW_SIGMA, SOFT_GLOW_WEIGHT)
    combined = np.maximum(tight, soft)

    screen = 255 - ((255 - arr) * (255 - combined) / 255)
    out = np.clip(screen, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(frame_path)


def encode_video(frame_dir, output_path, fps):
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path
    ], check=True)


def main():
    canvas_w, canvas_h = CANVAS_SIZE
    fly_frames = int(FLY_IN_SECONDS * FPS)
    hold_frames = int(HOLD_SECONDS * FPS)

    print("Loading image and building edge/fill/sparkle particle sets...")
    (positions, colors, sizes, max_alpha, halo_alpha, twinkle_amp,
     is_stationary) = load_particle_sets(SOURCE_IMAGE, canvas_w, canvas_h)
    print(f"Total particles: {len(positions)}")

    print("Rendering frames...")
    render_frames(positions, colors, sizes, max_alpha, halo_alpha, twinkle_amp,
                  is_stationary, canvas_w, canvas_h, fly_frames, hold_frames,
                  STAGGER, HALO_LAYER, HALO_SIZE_MULT, GLOW, BG_COLOR, RANDOM_SEED)

    print("Encoding video...")
    encode_video(FRAME_DIR, OUTPUT_VIDEO, FPS)

    shutil.rmtree(FRAME_DIR)
    print(f"Done! Saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()