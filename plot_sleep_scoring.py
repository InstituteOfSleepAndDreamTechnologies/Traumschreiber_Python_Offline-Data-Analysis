#!/usr/bin/env python3
"""
Sleep Staging with AnySleep – Automatic EEG-based sleep phase classification.

Uses the pre-trained AnySleep model to classify 30-second epochs of a full-night
EEG recording into Wake, N1, N2, N3, or REM sleep stages. Reads Traumschreiber
BDF (24 bit, current firmware) and EDF (16 bit, older recordings) files straight
from the microSD card. Outputs a visualization with probability panel
(hypnodensity), hypnogram, EEG spectrogram, battery (%) + ambient light, a raw
EEG viewer and sleep statistics against population norms.

Usage:
  python plot_sleep_scoring.py recording.BDF --channel 0
  python plot_sleep_scoring.py recording.BDF --channel 0 --start 3600 --duration 7200

Interactive plot: click a panel to load the raw-EEG window at the bottom, then
  ← / →          scroll in 10 s steps (shift: one full window)
  ↑ / ↓          amplitude scale, ±25 µV … ±1000 µV
  + / −          window width, 10/20/30/45/60/90 s
"""

import argparse
import struct
import urllib.request
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.special
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
from matplotlib.ticker import MultipleLocator

from anysleep_no_hydra import AnySleep


# Dark theme colors
DARK = '#0d1c2b'
WHITE = '#ffffff'
TEXT = '#aabbcc'
GRID = '#1a2f42'
ORANGE = '#ff9151'
GREEN = '#7be0ad'
BLUE = '#abd5ff'
CYAN = '#4dd4f4'

# Raw-signal panel — deliberately inverted against the dark theme: a hairline
# black trace on white reads like clinical paper EEG. Only the plot area is
# light; tick labels and axis labels sit outside it on the dark figure and keep
# the theme colors. ORANGE is too pale on white, hence RAW_ACCENT.
RAW_BG = '#ffffff'
RAW_INK = '#000000'
RAW_ACCENT = '#c94f00'

# Stage information (Wake=0, N1=1, N2=2, N3=3, REM=4)
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = [
    '#abd5ff',  # Wake
    '#fff4a8',  # N1  
    '#ffc8a8',  # N2  
    '#ff9151',  # N3  
    '#d151ff',  # REM 
]

# Hypnogram Y-axis: traditional order — Wake top (4), deep sleep bottom (0)
STAGE_TO_HYPO_Y = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}
HYPO_Y_LABELS = ['N3', 'N2', 'N1', 'REM', 'WAKE']

# Model weights auto-download
MODEL_URL = 'https://media.githubusercontent.com/media/dslaborg/AnySleep/main/models/anysleep-run1.pth'


# ── EDF reader ────────────────────────────────────────────────────────────────

def _read_chan_samples(raw, pos, n, bps):
    """Read n samples of one channel starting at byte `pos`.

    bps=2 → EDF int16 (little-endian). bps=3 → BDF int24 (little-endian,
    two's complement, sign-extended). Returns a sequence of Python/NumPy ints.
    """
    if bps == 2:
        return struct.unpack_from(f'<{n}h', raw, pos)
    a = np.frombuffer(raw, dtype=np.uint8, count=3 * n, offset=pos)
    a = a.reshape(n, 3).astype(np.int32)
    v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    return np.where(v & 0x800000, v - 0x1000000, v)


def _bdf_bps(raw):
    """3 bytes/sample for BDF (version byte 0xFF), else 2 for EDF."""
    return 3 if (len(raw) > 0 and raw[0] == 0xFF) else 2


def read_edf_channel(filepath, channel_idx):
    """Read one ADS channel (0–7) from a Traumschreiber EDF file.
    Returns (signal: np.ndarray float64, sample_rate: float, uv_per_lsb: float).

    `signal` holds the raw digital int16 sample values (no physical scaling
    applied). `uv_per_lsb` is the EDF affine slope
    (phys_max − phys_min) / (dig_max − dig_min) for the selected channel —
    multiply a digital value by it to get microvolts. For the Traumschreiber
    ADS131M08 chain (gain ×4, 24-bit BDF) this is (300000 − −300000) / (8388607 − −8388608) = 600000 / 16777215
    ≈ 0.03576 µV per digital unit. The derivation (ADC reference, PGA gain,
    header fields) is in the README of
    github.com/sd20-org/Traumschreiber_Python_Offline-Data-Analysis.
    """
    raw = Path(filepath).read_bytes()
    if len(raw) < 256:
        raise ValueError('File too small – not a valid EDF.')

    hdr = raw[:256].decode('latin-1')

    def _i(s):
        try: return int(s.strip())
        except ValueError: return 0

    def _f(s):
        try: return float(s.strip())
        except ValueError: return 1.0

    header_bytes    = _i(hdr[184:192]) or 3072
    num_records     = _i(hdr[236:244])      # -1 = unknown
    record_duration = _f(hdr[244:252])      # seconds per record
    num_signals     = _i(hdr[252:256]) or 11

    if not (0 <= channel_idx < min(num_signals, 8)):
        raise ValueError(
            f'Channel {channel_idx} invalid. Available EEG channels: '
            f'0–{min(num_signals, 8) - 1}.'
        )

    # EDF signal headers are stored as grouped fields across all signals:
    # [all labels (16 bytes each)] [all transducers (80 bytes each)] ...
    sig_hdr = raw[256:header_bytes].decode('latin-1')
    field_widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    field_names  = ['label', 'transducer', 'dim', 'phys_min', 'phys_max',
                    'dig_min', 'dig_max', 'prefilter', 'num_samples', 'reserved']
    sigs = [{} for _ in range(num_signals)]
    off = 0
    for fw, fn in zip(field_widths, field_names):
        for i in range(num_signals):
            sigs[i][fn] = sig_hdr[off:off + fw].strip()
            off += fw

    ns = []
    for s in sigs:
        try: ns.append(int(s['num_samples']))
        except ValueError: ns.append(0)

    bps = _bdf_bps(raw)
    record_bytes = sum(n * bps for n in ns)
    if record_bytes == 0:
        raise ValueError('EDF header reports 0 samples/record – file corrupt?')

    sample_rate = ns[0] / record_duration if record_duration > 0 else 250.0

    # Collect raw digital samples for channel_idx across all data records
    samples = []
    offset = header_bytes
    rec = 0
    while offset + record_bytes <= len(raw):
        if num_records >= 0 and rec >= num_records:
            break
        pos = offset
        for i, n in enumerate(ns):
            if i == channel_idx:
                samples.extend(_read_chan_samples(raw, pos, n, bps))
            pos += n * bps
        offset = pos
        rec += 1

    # EDF physical scaling for the selected channel. The Traumschreiber
    # firmware writes phys_min/max = ∓300000 µV (gain ×4) and dig_min/max = ∓8388608/8388607
    # for every ADS channel; the slope below converts digital → µV.
    sel = sigs[channel_idx]
    try:
        phys_min = float(sel['phys_min']); phys_max = float(sel['phys_max'])
        dig_min  = float(sel['dig_min']);  dig_max  = float(sel['dig_max'])
        uv_per_lsb = (phys_max - phys_min) / (dig_max - dig_min)
        if not np.isfinite(uv_per_lsb) or uv_per_lsb <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError, KeyError):
        uv_per_lsb = 300000.0 / 8388608.0  # firmware default at gain ×4 (±300000 µV ↔ int24)

    return np.array(samples, dtype=np.float64), float(sample_rate), float(uv_per_lsb)


def battery_mv_to_percent(mv):
    """Battery voltage → percent, the same linear curve as the firmware
    (battery_monitor.c) and the sd20 App: 2500 mV = 0 %, 2800 mV = 100 %."""
    return np.clip((np.asarray(mv, dtype=np.float64) - 2500.0) * 100.0 / 300.0,
                   0.0, 100.0)


def read_edf_aux(filepath, labels=('Battery', 'Light')):
    """Read the slow auxiliary signals (1 sample/record) by label.

    Returns {label: (t_s, values_phys)} for every label found — t_s are
    seconds from recording start, values scaled with the header's affine
    (mV for the Traumschreiber's Battery and Light). Missing labels (older
    firmware without aux signals) are simply absent from the dict.
    """
    raw = Path(filepath).read_bytes()
    hdr = raw[:256].decode('latin-1')
    header_bytes = int(hdr[184:192].strip() or 0) or 3072
    num_records = int(hdr[236:244].strip() or -1)
    record_duration = float(hdr[244:252].strip() or 1.0)
    ns_sig = int(hdr[252:256].strip() or 0)
    sh = raw[256:header_bytes].decode('latin-1')

    def field(off_w, w):
        base = off_w * ns_sig
        return [sh[base + i * w: base + (i + 1) * w].strip() for i in range(ns_sig)]

    # Field offsets in units of ns_sig bytes: label 16, transducer 80, dim 8,
    # pmin 8, pmax 8, dmin 8, dmax 8, prefilter 80, num_samples 8.
    lab = field(0, 16)
    pmin, pmax = field(104, 8), field(112, 8)
    dmin, dmax = field(120, 8), field(128, 8)
    nsamp = [int(x or 0) for x in field(216, 8)]

    bps = _bdf_bps(raw)
    record_bytes = sum(nsamp) * bps
    n_rec = (len(raw) - header_bytes) // record_bytes
    if num_records >= 0:
        n_rec = min(n_rec, num_records)
    if n_rec <= 0:
        return {}
    recs = np.frombuffer(raw, dtype=np.uint8, count=n_rec * record_bytes,
                         offset=header_bytes).reshape(n_rec, record_bytes)

    out = {}
    for want in labels:
        if want not in lab:
            continue
        i = lab.index(want)
        n = nsamp[i]
        pos = sum(nsamp[:i]) * bps
        a = recs[:, pos:pos + n * bps].reshape(-1, bps).astype(np.int32)
        if bps == 3:
            v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
        else:
            v = (a[:, 0] | (a[:, 1] << 8)).astype(np.int16).astype(np.int32)
        try:
            k = (float(pmax[i]) - float(pmin[i])) / (float(dmax[i]) - float(dmin[i]))
            phys = (v - float(dmin[i])) * k + float(pmin[i])
        except (ValueError, ZeroDivisionError):
            phys = v.astype(np.float64)
        t_s = np.arange(len(v)) * (record_duration / n)
        out[want] = (t_s, phys)
    return out


# ── Model helpers ─────────────────────────────────────────────────────────────

def ensure_model(model_path):
    """Download AnySleep weights if not present."""
    model_path = Path(model_path)
    if model_path.exists():
        return str(model_path)

    print(f'Model not found: {model_path}')
    print('Downloading AnySleep model (~12 MB)...')
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f'✓ Model saved to {model_path}')
    except Exception as e:
        raise RuntimeError(f'Download failed: {e}\nManually download from:\n{MODEL_URL}')
    return str(model_path)


# ── Signal preprocessing ──────────────────────────────────────────────────────

def preprocess(signal, original_fs, verbose=False):
    """Filter, resample to 128 Hz, clip, robust scale — for AnySleep input.

    AnySleep was trained on clinical PSG signals, which are hardware-filtered
    by the amplifier. The Traumschreiber records broadband — without explicit
    filtering, DC drift and powerline noise dominate the RobustScaler statistics
    and the model collapses to WAKE. We apply a clinical-style bandpass first.

    Returns (sig_model, sig_spec):
      sig_model : notch + BP 0.3–35 Hz + resample + clip + scale → model input
      sig_spec  : notch + HP 0.3 Hz   + resample                  → spectrogram
    """
    sig = np.asarray(signal, dtype=np.float64)

    if verbose:
        print(f'  raw    : median={np.median(sig):+9.1f}  iqr={np.subtract(*np.percentile(sig, [75, 25])):.1f}  range=[{sig.min():.0f}, {sig.max():.0f}]')

    # 50 Hz notch (Europe powerline) — applied to both paths
    b_notch, a_notch = scipy.signal.iirnotch(50.0, 30.0, fs=original_fs)
    sos_notch = scipy.signal.tf2sos(b_notch, a_notch)
    sig_n = scipy.signal.sosfiltfilt(sos_notch, sig)

    # ── Spectrogram path: HP 0.3 Hz only (keeps content up to Nyquist) ──────
    sos_hp = scipy.signal.butter(4, 0.3, btype='highpass',
                                  fs=original_fs, output='sos')
    sig_hp = scipy.signal.sosfiltfilt(sos_hp, sig_n)
    sig_spec = scipy.signal.resample_poly(sig_hp, 128, int(original_fs))

    # ── Model path: full BP 0.3–35 Hz to match training distribution ────────
    sos_bp = scipy.signal.butter(4, [0.3, 35.0], btype='bandpass',
                                  fs=original_fs, output='sos')
    sig_bp = scipy.signal.sosfiltfilt(sos_bp, sig_n)

    if verbose:
        print(f'  filt   : median={np.median(sig_bp):+9.3f}  iqr={np.subtract(*np.percentile(sig_bp, [75, 25])):.3f}  range=[{sig_bp.min():.1f}, {sig_bp.max():.1f}]')

    sig_m = scipy.signal.resample_poly(sig_bp, 128, int(original_fs))

    iqr = float(np.subtract(*np.percentile(sig_m, [75, 25])))
    if iqr > 0:
        sig_m = np.clip(sig_m, -20 * iqr, 20 * iqr)

    median = float(np.median(sig_m))
    iqr = float(np.subtract(*np.percentile(sig_m, [75, 25])))
    sig_m = (sig_m - median) / max(iqr, 1e-8)

    if verbose:
        print(f'  scaled : median={np.median(sig_m):+9.3f}  iqr={np.subtract(*np.percentile(sig_m, [75, 25])):.3f}  range=[{sig_m.min():.2f}, {sig_m.max():.2f}]')

    return sig_m, sig_spec


def run_inference(signal, fs, model_path, device='cpu', verbose=True,
                  model_type='pth'):
    """Preprocess, run model, return (preds, probs, n_epochs, sig_128hz).

    `model_type`:
      * `'pth'`  — load AnySleep PyTorch weights directly (reference path).
        `model_path` is a `.pth` file.
      * `'onnx'` — load an ONNX export and infer via onnxruntime
        (matches the on-device path used by the Flutter app). `model_path`
        is a `.onnx` file. Useful for verifying that the re-exported
        branchless ONNX in the phone app produces the same predictions as
        the PyTorch reference, before debating why the on-device run
        differs from the python run.
    """
    sig_scaled, sig_128 = preprocess(signal, fs, verbose=verbose)

    epoch_samples = 128 * 30  # 3840
    n_epochs = len(sig_scaled) // epoch_samples
    if n_epochs == 0:
        raise ValueError('Signal too short for even one 30-second epoch.')

    sig_trimmed = sig_scaled[:n_epochs * epoch_samples]

    if model_type == 'pth':
        model = AnySleep(path=model_path, sleep_stage_frequency=1)
        model = model.to(device).eval()

        # Shape: (1, total_samples, 1) — single channel
        x = torch.from_numpy(sig_trimmed.reshape(1, -1, 1)).float().to(device)
        with torch.no_grad():
            logits = model(x)  # (1, n_epochs, 5)
        logits_np = logits.cpu().numpy()[0]                    # (n_epochs, 5)

    elif model_type == 'onnx':
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(model_path), providers=['CPUExecutionProvider'])
        x_np = sig_trimmed.reshape(1, -1, 1).astype(np.float32)
        logits = sess.run(None, {'signal': x_np})[0]            # (1, n_epochs, 5)
        logits_np = logits[0]

    else:
        raise ValueError(
            f"Unknown model_type={model_type!r}; expected 'pth' or 'onnx'.")

    probs = scipy.special.softmax(logits_np, axis=1)           # (n_epochs, 5)
    preds = np.argmax(probs, axis=1)

    # Trim sig_128 to match epoch count
    sig_128 = sig_128[:n_epochs * epoch_samples]

    return preds, probs, n_epochs, sig_128


# ── Plotting ──────────────────────────────────────────────────────────────────

def compute_spectrogram(signal, fs=128, fmax=30, median_t=31.0, median_f=3.0):
    """High-resolution spectrogram with 2D median-filter smoothing.

    Window 4 s, 75% overlap → 1 s per time bin, 0.25 Hz per freq bin.
    Median filter window: median_t seconds × median_f Hz.
    """
    nperseg  = int(fs * 4)           # 4-second window
    noverlap = int(nperseg * 0.75)   # 75% overlap → 1 s/bin
    f, t, Sxx = scipy.signal.spectrogram(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density'
    )
    mask = f <= fmax
    f = f[mask]
    logP = np.log10(Sxx[mask, :] + 1e-20)

    # 2D median filter (preserves edges in stage transitions)
    if median_t > 0 and median_f > 0:
        dt_spec = float(t[1] - t[0]) if len(t) > 1 else 1.0
        df_spec = float(f[1] - f[0]) if len(f) > 1 else 0.25

        def _odd(n): return n if n % 2 else n + 1
        med_t = max(3, _odd(int(round(median_t / dt_spec))))
        med_f = max(1, _odd(int(round(median_f / df_spec))))
        logP = scipy.ndimage.median_filter(logP, size=(med_f, med_t), mode='reflect')

    return f, t, logP


def _fmt_hm(seconds):
    """Format duration as 'H:MM' (e.g. '1:23')."""
    if seconds is None:
        return '—'
    h = int(seconds // 3600)
    m = int(round((seconds % 3600) / 60))
    if m == 60:
        h += 1
        m = 0
    return f'{h}:{m:02d}'


# Normative values for healthy adults (~25–45 y)
# Sources: Ohayon et al. 2004 (meta-analysis); Boulos et al. 2019 (Lancet
# Respir Med); AASM scoring manual. Means and SDs are pooled across studies.
NORMS = {
    'TIB':  {'mean': 8.0,  'sd': 1.0,  'unit': 'h',   'in_s': True,  'lo': 0},
    'TST':  {'mean': 7.0,  'sd': 1.0,  'unit': 'h',   'in_s': True,  'lo': 0},
    'SOL':  {'mean': 15.0, 'sd': 12.0, 'unit': 'min', 'in_s': True,  'lo': 0},
    'WASO': {'mean': 30.0, 'sd': 25.0, 'unit': 'min', 'in_s': True,  'lo': 0},
    'SE':   {'mean': 87.0, 'sd': 8.0,  'unit': '%',   'in_s': False, 'lo': 0, 'hi': 100},
    'RL':   {'mean': 85.0, 'sd': 35.0, 'unit': 'min', 'in_s': True,  'lo': 0},
    'AW':   {'mean': 8.0,  'sd': 5.0,  'unit': '',    'in_s': False, 'lo': 0},
    # Stage proportions (% of TIB for Wake, % of TST for sleep stages)
    'Wake': {'mean': 10.0, 'sd': 6.0,  'unit': '%',   'in_s': False, 'lo': 0, 'hi': 100},
    'N1':   {'mean': 6.0,  'sd': 3.0,  'unit': '%',   'in_s': False, 'lo': 0, 'hi': 100},
    'N2':   {'mean': 50.0, 'sd': 7.0,  'unit': '%',   'in_s': False, 'lo': 0, 'hi': 100},
    'N3':   {'mean': 17.0, 'sd': 8.0,  'unit': '%',   'in_s': False, 'lo': 0, 'hi': 100},
    'REM':  {'mean': 22.0, 'sd': 5.0,  'unit': '%',   'in_s': False, 'lo': 0, 'hi': 100},
}


def _to_norm_unit(value, key):
    """Convert raw value (seconds, %, count) to the norm's display unit."""
    if value is None or key not in NORMS:
        return None
    n = NORMS[key]
    if n['unit'] == 'h':
        return value / 3600.0 if n['in_s'] else value
    if n['unit'] == 'min':
        return value / 60.0 if n['in_s'] else value
    return value  # %, count


def _fmt_norm(key):
    """Format the norm range as 'mean ± sd unit'."""
    n = NORMS[key]
    if n['unit'] == 'h':
        return f"{n['mean']:.0f}±{n['sd']:.0f} h"
    if n['unit'] == 'min':
        return f"{n['mean']:.0f}±{n['sd']:.0f} min"
    if n['unit'] == '%':
        return f"{n['mean']:.0f}±{n['sd']:.0f} %"
    return f"{n['mean']:.0f}±{n['sd']:.0f}"


def _draw_minibox(ax, x0, x1, y, value_raw, key, h_box=0.022):
    """Horizontal mini boxplot at row y showing population norm + user value.

    Box   : μ ± 1·SD   (typical range, ~68% of population)
    Whisker: μ ± 2.5·SD (extended normal, ~99%)
    Marker: user's measured value (red if outside whisker range)
    """
    if value_raw is None or key not in NORMS:
        return
    n = NORMS[key]
    value = _to_norm_unit(value_raw, key)
    mean, sd = n['mean'], n['sd']

    range_lo = max(n.get('lo', -1e9), mean - 2.5 * sd)
    range_hi = min(n.get('hi',  1e9), mean + 2.5 * sd)
    width = x1 - x0

    def nx(v):
        v_c = np.clip((v - range_lo) / (range_hi - range_lo), 0, 1)
        return x0 + v_c * width

    whisker_color = '#90a5b8'
    box_face      = '#7a9ab0'
    mean_color    = '#ffffff'
    user_good     = '#4ade80'   # green   — within ±1σ
    user_meh      = '#ffe851'   # yellow  — within ±2.5σ, outside ±1σ
    user_bad      = '#ff4466'   # red     — outside ±2.5σ

    # Whisker line
    ax.plot([x0, x1], [y, y], color=whisker_color, lw=1.1,
            transform=ax.transAxes, clip_on=False, solid_capstyle='butt')
    # Whisker caps
    cap = 0.017
    for xx in (x0, x1):
        ax.plot([xx, xx], [y - cap, y + cap], color=whisker_color, lw=1.1,
                transform=ax.transAxes, clip_on=False)

    # Box (μ ± σ)
    bx0, bx1 = nx(mean - sd), nx(mean + sd)
    box = plt.Rectangle((bx0, y - h_box), bx1 - bx0, 2 * h_box,
                        facecolor=box_face, alpha=0.55, edgecolor=whisker_color,
                        linewidth=0.8,
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    # Mean tick
    ax.plot([nx(mean), nx(mean)], [y - h_box * 1.35, y + h_box * 1.35],
            color=mean_color, lw=1.8, transform=ax.transAxes, clip_on=False)

    # User marker — three-tier color: green/yellow/red
    dev = abs(value - mean)
    if dev <= sd:
        user_color = user_good
    elif dev <= 2.5 * sd:
        user_color = user_meh
    else:
        user_color = user_bad

    is_out = (value < range_lo) or (value > range_hi)
    marker_x = nx(value)
    ax.plot(marker_x, y, 'o', markersize=9, color=user_color,
            markeredgecolor=DARK, markeredgewidth=1.4,
            transform=ax.transAxes, clip_on=False, zorder=10)
    if is_out:
        arrow_x = x1 + 0.008 if value > range_hi else x0 - 0.008
        arrow_marker = '▶' if value > range_hi else '◀'
        ax.text(arrow_x, y, arrow_marker, color=user_bad, fontsize=9,
                fontweight='800',
                transform=ax.transAxes, va='center',
                ha='left' if value > range_hi else 'right')


def compute_sleep_stats(preds, epoch_s=30):
    """Classical sleep architecture metrics from epoch-level predictions.

    Conventions:
      - Sleep onset  = first non-Wake epoch
      - TST          = non-Wake epochs from onset to end of recording
      - WASO         = Wake epochs after sleep onset
      - SE           = TST / TIB
      - REM latency  = onset → first REM epoch
      - Awakenings   = sleep→wake transitions after onset
      - Stage %      = % of TST for N1/N2/N3/REM; % of TIB for Wake
    """
    preds = np.asarray(preds)
    n = len(preds)
    tib_s = n * epoch_s

    sleep_mask = preds != 0          # 0 = Wake
    if not sleep_mask.any():
        return {
            'tib_s': tib_s, 'tst_s': 0, 'sol_s': tib_s,
            'waso_s': 0, 'se_pct': 0.0, 'rem_lat_s': None,
            'n_awakenings': 0,
            'stage_times': {n: 0 for n in STAGE_NAMES},
            'stage_pct':   {n: 0.0 for n in STAGE_NAMES},
        }

    onset_idx = int(np.argmax(sleep_mask))
    sol_s = onset_idx * epoch_s

    after = preds[onset_idx:]
    tst_epochs  = int(np.sum(after != 0))
    waso_epochs = int(np.sum(after == 0))
    tst_s  = tst_epochs * epoch_s
    waso_s = waso_epochs * epoch_s
    se_pct = 100.0 * tst_s / tib_s if tib_s else 0.0

    rem_mask_after = after == 4
    rem_lat_s = int(np.argmax(rem_mask_after)) * epoch_s if rem_mask_after.any() else None

    sleep_to_wake = (after[:-1] != 0) & (after[1:] == 0)
    n_awakenings  = int(np.sum(sleep_to_wake))

    stage_times, stage_pct = {}, {}
    for i, name in enumerate(STAGE_NAMES):
        count = int(np.sum(preds == i))
        stage_times[name] = count * epoch_s
        denom = n if name == 'Wake' else max(tst_epochs, 1)
        stage_pct[name] = 100.0 * count / denom

    return {
        'tib_s': tib_s, 'tst_s': tst_s, 'sol_s': sol_s,
        'waso_s': waso_s, 'se_pct': se_pct, 'rem_lat_s': rem_lat_s,
        'n_awakenings': n_awakenings,
        'stage_times': stage_times, 'stage_pct': stage_pct,
    }


def _add_stats_panel(fig, gs_pos, stats):
    """Sleep statistics panel: metrics + normative boxplots."""
    ax = fig.add_subplot(gs_pos)
    ax.set_facecolor(DARK)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3f52')
        spine.set_linewidth(1)

    # ── Section headers ──────────────────────────────────────────────────────
    ax.text(0.008, 0.93, 'SLEEP STATISTICS', color=WHITE,
            fontsize=11, fontweight='800', transform=ax.transAxes, va='top')
    ax.text(0.515, 0.93, 'STAGE BREAKDOWN', color=WHITE,
            fontsize=11, fontweight='800', transform=ax.transAxes, va='top')

    # Legend explainer — single line, right-aligned
    legend_y = 0.91
    # First the descriptive text on the left of the legend area
    ax.text(0.745, legend_y,
            'box ±1σ · whisker ±2.5σ ·',
            color='#7a9ab0', fontsize=8, fontstyle='italic',
            transform=ax.transAxes, va='center', ha='left')
    # Then three color dots with their meaning, inline
    for x_dot, x_lbl, label, col in (
            (0.870, 0.882, 'in ±1σ',    '#4ade80'),
            (0.916, 0.927, 'in ±2.5σ',  '#ffe851'),
            (0.974, 0.985, 'out',       '#ff4466'),
    ):
        ax.plot(x_dot, legend_y, 'o', markersize=7,
                color=col, markeredgecolor=DARK, markeredgewidth=1.0,
                transform=ax.transAxes, clip_on=False)
        ax.text(x_lbl, legend_y, label,
                color='#7a9ab0', fontsize=8,
                transform=ax.transAxes, va='center', ha='left')

    # ── LEFT: overall sleep metrics ──────────────────────────────────────────
    overall = [
        ('Time in bed (TIB)',          _fmt_hm(stats['tib_s']),      stats['tib_s'],     'TIB'),
        ('Total sleep time (TST)',     _fmt_hm(stats['tst_s']),      stats['tst_s'],     'TST'),
        ('Sleep onset latency (SOL)',  _fmt_hm(stats['sol_s']),      stats['sol_s'],     'SOL'),
        ('Wake after onset (WASO)',    _fmt_hm(stats['waso_s']),     stats['waso_s'],    'WASO'),
        ('Sleep efficiency (SE)',      f"{stats['se_pct']:.1f} %",   stats['se_pct'],    'SE'),
        ('REM latency (RL)',           _fmt_hm(stats['rem_lat_s']),  stats['rem_lat_s'], 'RL'),
        ('Awakenings (AW)',            str(stats['n_awakenings']),   stats['n_awakenings'], 'AW'),
    ]

    # Left columns
    y0_l   = 0.78
    dy_l   = 0.105
    x_lbl  = 0.008
    x_val  = 0.235     # right-aligned numeric value
    box_x0 = 0.26
    box_x1 = 0.40
    x_norm = 0.415     # norm range text starts here

    for i, (label, val_str, val_raw, key) in enumerate(overall):
        y = y0_l - i * dy_l
        ax.text(x_lbl, y, label, color=TEXT, fontsize=10,
                transform=ax.transAxes, va='center')
        ax.text(x_val, y, val_str, color='white', fontsize=11,
                fontweight='700', family='monospace',
                transform=ax.transAxes, va='center', ha='right')
        _draw_minibox(ax, box_x0, box_x1, y, val_raw, key)
        ax.text(x_norm, y, _fmt_norm(key),
                color='#7a9ab0', fontsize=8.5, family='monospace',
                transform=ax.transAxes, va='center')

    # ── RIGHT: per-stage breakdown ───────────────────────────────────────────
    # 5 rows, distributed over same height as 7 rows on the left
    y0_r   = 0.78
    dy_r   = dy_l * 7 / 5 * 0.9   # similar vertical span, comfortable spacing
    x_sw    = 0.520
    x_name  = 0.545
    x_time  = 0.612      # right-aligned
    x_pct   = 0.680      # right-aligned (% of TST or TIB)
    bx0     = 0.700
    bx1     = 0.870
    x_norm_r = 0.880

    for i, name in enumerate(STAGE_NAMES):
        y = y0_r - i * dy_r
        color = STAGE_COLORS[i]
        # Color swatch
        ax.add_patch(plt.Rectangle((x_sw, y - 0.038), 0.016, 0.076,
                                    color=color, alpha=0.9,
                                    transform=ax.transAxes, clip_on=False))
        ax.text(x_name, y, name, color='white', fontsize=11,
                fontweight='700', transform=ax.transAxes, va='center')
        ax.text(x_time, y, _fmt_hm(stats['stage_times'][name]),
                color='white', fontsize=11, fontweight='700',
                family='monospace',
                transform=ax.transAxes, va='center', ha='right')
        denom = 'TIB' if name == 'Wake' else 'TST'
        ax.text(x_pct, y, f"{stats['stage_pct'][name]:.1f} %",
                color='white', fontsize=11, fontweight='700',
                family='monospace',
                transform=ax.transAxes, va='center', ha='right')

        _draw_minibox(ax, bx0, bx1, y, stats['stage_pct'][name], name)
        ax.text(x_norm_r, y, _fmt_norm(name) + f' ({denom})',
                color='#7a9ab0', fontsize=8.5, family='monospace',
                transform=ax.transAxes, va='center')


def _style_ax(ax):
    ax.set_facecolor(DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3f52')
        spine.set_linewidth(1)
    ax.tick_params(colors=TEXT, labelsize=9)


def plot_sleep_staging(preds, probs, n_epochs, sig_128, ch_label='', fmax=30.0,
                       uv_per_lsb=300000.0 / 8388608.0, fs_128=128.0, aux=None):
    """4-panel interactive figure:
    probabilities | hypnogram | EEG spectrogram | raw EEG (click + arrow keys).

    Click anywhere on the probability, hypnogram, or spectrogram panel to
    render the raw-EEG window (filtered like the spectrogram path, scaled to
    µV via `uv_per_lsb`) in the bottom signal panel. ← / → then scroll that
    window in 10 s steps (shift: a full window per press), ↑ / ↓ step the
    amplitude scale through RAW_YLIMS and + / − the width through RAW_WINDOWS.

    `aux` is the dict from read_edf_aux(), time-aligned to the analysed span.
    When it holds Battery and/or Light, a slim panel below the spectrogram
    shows both (clickable like the panels above); without it the layout is
    the plain 4-panel one.
    """
    duration_h = n_epochs * 30 / 3600
    t_h = np.arange(n_epochs) * 30 / 3600        # epoch start times in hours

    has_aux = bool(aux)
    fig = plt.figure(figsize=(20, 14.5 if has_aux else 13.5), facecolor=DARK)
    # The three upper panels are 20 % shorter than before (1.2→0.96,
    # 0.8→0.64, 1.5→1.2) to make room for the new raw-signal panel.
    # The optional battery/light panel sits right under the spectrogram so it
    # shares the hour axis with the panels above it.
    if has_aux:
        gs = gridspec.GridSpec(6, 2, height_ratios=[0.96, 0.64, 1.2, 0.45, 1.0, 1.5],
                                width_ratios=[80, 1], hspace=0.42, wspace=0.012)
        ROW_AUX, ROW_RAW, ROW_STATS = 3, 4, 5
    else:
        gs = gridspec.GridSpec(5, 2, height_ratios=[0.96, 0.64, 1.2, 1.0, 1.5],
                                width_ratios=[80, 1], hspace=0.42, wspace=0.012)
        ROW_AUX, ROW_RAW, ROW_STATS = None, 3, 4

    # ── Panel 0: stacked probability area ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.stackplot(t_h, *[probs[:, i] for i in range(5)],
                  labels=STAGE_NAMES, colors=STAGE_COLORS, alpha=0.85)
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, duration_h)
    ax0.set_ylabel('Probability', color=TEXT, fontsize=11, fontweight='600')
    ax0.legend(loc='upper right', framealpha=0.85, fontsize=10,
               labelcolor=DARK, facecolor=TEXT)
    ax0.grid(True, alpha=0.15, color=GRID, linestyle='--', linewidth=0.5)
    _style_ax(ax0)
    ax0.set_xticklabels([])

    # ── Panel 1: hypnogram ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    hypo_y = np.array([STAGE_TO_HYPO_Y[int(p)] for p in preds])

    # Colored fill per segment — thin band centred on the stage level
    # (was ±0.45 ≈ 1 cm tall; now ±0.09 ≈ 0.2 cm so the stages read as a
    # crisp step trace rather than fat blocks).
    for i in range(n_epochs):
        x0 = t_h[i]
        x1 = t_h[i] + 30 / 3600
        y  = hypo_y[i]
        ax1.fill_between([x0, x1], y - 0.09, y + 0.09,
                         color=STAGE_COLORS[int(preds[i])], alpha=0.85)

    ax1.step(t_h, hypo_y, where='post', color='white', linewidth=0.6, alpha=0.6)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(HYPO_Y_LABELS)
    ax1.set_ylim(-0.6, 4.6)
    ax1.set_xlim(0, duration_h)
    ax1.set_ylabel('Stage', color=TEXT, fontsize=11, fontweight='600')
    ax1.grid(True, alpha=0.12, color=GRID, linestyle='--', linewidth=0.5, axis='x')
    _style_ax(ax1)
    ax1.set_xticklabels([])

    # ── Panel 2: EEG spectrogram ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    f, t_spec, logP = compute_spectrogram(sig_128, fs=128, fmax=fmax)

    # Median-anchored color scale (instead of percentile[5,97]): robust to
    # montage-dependent noise floors. A percentile range is dragged down by a
    # very deep floor — e.g. a bipolar derivation's 35–60 Hz floor — which
    # stretches the scale and washes the panel out / over-reddens it. Anchoring
    # to the median keeps the *typical* power at a consistent blue-green and
    # shows a fixed ~3-decade window, so different recordings/montages (TS vs
    # clinical PSG) render comparably at any fmax. For a typical TS EEG channel
    # this reproduces the previous percentile[5,97] look almost exactly.
    _med = float(np.median(logP))
    vmin, vmax = _med - 0.7, _med + 2.3
    im = ax2.pcolormesh(t_spec / 3600, f, logP,
                        shading='auto', cmap='turbo',
                        vmin=vmin, vmax=vmax, rasterized=True)
    ax2.set_ylim(0, fmax)
    ax2.set_xlim(0, duration_h)
    ax2.set_ylabel('Frequency (Hz)', color=TEXT, fontsize=11, fontweight='600')
    _style_ax(ax2)
    if has_aux:
        ax2.set_xticklabels([])
    else:
        ax2.set_xlabel('Time (hours)', color=TEXT, fontsize=11, fontweight='600')

    cax = fig.add_subplot(gs[2, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('log₁₀(Power)', color=TEXT, fontsize=9)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.outline.set_edgecolor('#2a3f52')

    # ── Panel 2b: battery + ambient light (optional) ─────────────────────────
    # Battery on the left axis (percent, as the app shows it), light on a twin
    # right axis. The Traumschreiber light sensor reads HIGH when it is dark
    # (≈3.3 V at night, tens of mV in daylight), so it is plotted negated.
    ax_aux = None
    _aux_axes = ()
    if has_aux:
        ax_aux = fig.add_subplot(gs[ROW_AUX, 0])
        _style_ax(ax_aux)
        ax_aux.set_xlim(0, duration_h)
        ax_aux.grid(True, alpha=0.15, color=GRID, linestyle='--', linewidth=0.5)
        ax_aux.set_xlabel('Time (hours)', color=TEXT, fontsize=11, fontweight='600')
        handles = []
        if 'Battery' in aux:
            t_b, v_b = aux['Battery']
            # 1 Hz samples carry ~±10 mV ADC noise; a 60 s running mean shows
            # the discharge curve instead of a noise band.
            if len(v_b) > 60:
                v_b = scipy.ndimage.uniform_filter1d(v_b, 60, mode='nearest')
            h, = ax_aux.plot(t_b / 3600, battery_mv_to_percent(v_b), color=GREEN,
                             linewidth=1.2, label='Battery')
            ax_aux.set_ylabel('Battery (%)', color=GREEN, fontsize=10,
                              fontweight='600')
            ax_aux.set_ylim(0, 100)
            ax_aux.yaxis.set_major_locator(MultipleLocator(50))
            handles.append(h)
        if 'Light' in aux:
            t_l, v_l = aux['Light']
            ax_l = ax_aux.twinx()
            _style_ax(ax_l)
            ax_l.patch.set_visible(False)
            # Sign flipped so that up = brighter.
            h, = ax_l.plot(t_l / 3600, -v_l, color=BLUE, linewidth=0.9, alpha=0.9,
                           label='Light')
            ax_l.set_ylabel('Light (−mV)\nup = brighter', color=BLUE, fontsize=9,
                            fontweight='600')
            ax_l.set_xlim(0, duration_h)
            handles.append(h)
            _aux_axes = (ax_l,)
        ax_aux.legend(handles=handles, loc='upper left', framealpha=0.85,
                      fontsize=9, labelcolor=DARK, facecolor=TEXT)
        _aux_axes = (ax_aux,) + _aux_axes

    # ── Panel 3: raw EEG window (click + ← / → / ↑ / ↓ navigation) ───────────
    # Navigation is blit-based, not redraw-based. Panel 2 holds a full-night
    # pcolormesh (1000+ quads), so a `draw_idle()` per keypress costs ~2 s and
    # makes a held-down arrow key unusable.
    #
    # The markers on the three upper panels are individually `animated=True`.
    # ax_raw needs two cache levels, because its two interactions dirty
    # different amounts of pixels:
    #
    #   scroll (← / →)  moves only the trace — axis chrome is unchanged
    #   zoom   (↑ / ↓)  changes ylim, hence grid lines and y tick labels too
    #
    # So the whole axes is animated (absent from the figure background, so a
    # zoom never needs a figure redraw), and the trace plus stamp are animated
    # on top of that (absent from the axes' own draw). That yields a reusable
    # `bg_chrome` snapshot holding patch+grid+ticks: a scroll restores it and
    # draws just the trace, a zoom rebuilds it first. Redrawing chrome on every
    # scroll instead costs ~34 ms — right at typical key-repeat rate.
    #
    # The upgrade is deferred to the first navigation event so that a
    # non-interactive `--save` run renders ax_raw the ordinary way.
    RAW_STEP_S = 10.0     # ← / → step; shift+← / shift+→ jumps a full window
    # ↑ / ↓ step through these half-ranges (µV). 75 = the previous fixed scale.
    RAW_YLIMS = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0)
    RAW_YLIM_I0 = 2
    # + / − step through these window widths (s). 30 = the previous fixed width.
    RAW_WINDOWS = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
    RAW_WINDOW_I0 = 2

    # matplotlib binds left/right to its view-history (keymap.back/.forward) by
    # default, which would fire alongside our handler and yank the view around.
    # up/down and +/− are unbound in the default keymap, so they need no surgery.
    for _km in ('keymap.back', 'keymap.forward'):
        plt.rcParams[_km] = [k for k in plt.rcParams[_km]
                             if k not in ('left', 'right')]

    sig_dur_s = len(sig_128) / fs_128

    ax_raw = fig.add_subplot(gs[ROW_RAW, 0])
    _style_ax(ax_raw)
    ax_raw.set_facecolor(RAW_BG)         # white plot area, see RAW_BG comment
    ax_raw.set_ylim(-RAW_YLIMS[RAW_YLIM_I0], RAW_YLIMS[RAW_YLIM_I0])
    # x spans the current window width; absolute position, window width and
    # amplitude scale are all reported by raw_stamp.
    ax_raw.set_xlim(-RAW_WINDOWS[RAW_WINDOW_I0] / 2, RAW_WINDOWS[RAW_WINDOW_I0] / 2)
    ax_raw.set_ylabel('Amplitude (µV)', color=TEXT, fontsize=11, fontweight='600')
    ax_raw.set_xlabel('Time within window (s)  ·  0 = window centre',
                      color=TEXT, fontsize=11, fontweight='600')
    ax_raw.grid(True, alpha=0.18, color=RAW_INK, linestyle='--', linewidth=0.5)
    # One-second ruling, paper-EEG style. Minor gridlines rather than extra
    # axvlines: the locator refills them by itself when + / − change the width.
    # length=0 keeps the ruling free of tick marks, and minor ticks carry no
    # labels by default, so the major labels below stay uncluttered.
    ax_raw.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax_raw.tick_params(axis='x', which='minor', length=0)
    ax_raw.grid(True, which='minor', axis='x', alpha=0.16, color=RAW_INK,
                linewidth=0.4)
    ax_raw.axvline(0.0, color=RAW_ACCENT, linewidth=1.0, alpha=0.85)
    ax_raw.axhline(0.0, color=RAW_INK, linewidth=0.6, alpha=0.28)
    raw_hint = ax_raw.text(0.5, 0.5,
                'Click on any panel above (probability, hypnogram, spectrogram'
                + (', battery/light' if has_aux else '') + ')\n'
                'to view the raw EEG window here (filtered, µV).\n'
                '← / → scroll in 10 s steps (shift: full window),\n'
                '↑ / ↓ amplitude scale,  + / −  window width.',
                color='#55606b', fontsize=11, fontstyle='italic',
                ha='center', va='center', transform=ax_raw.transAxes)

    # Updated via set_data/set_text per step. Deliberately NOT animated: they
    # are children of ax_raw, which gets animated as a whole below — an
    # animated child would be skipped by its parent's draw().
    raw_line, = ax_raw.plot([], [], color=RAW_INK, linewidth=0.35)
    # Boxed: at high zoom the trace runs straight through this corner, and on
    # white an unbacked label becomes unreadable.
    raw_stamp = ax_raw.text(0.995, 0.955, '', transform=ax_raw.transAxes,
                            color=RAW_ACCENT, fontsize=10, family='monospace',
                            ha='right', va='top',
                            bbox=dict(boxstyle='square,pad=0.3', facecolor=RAW_BG,
                                      edgecolor='none', alpha=0.85))
    _markers = [axu.axvline(0.0, color=ORANGE, linewidth=1.0, alpha=0.7,
                            animated=True, visible=False)
                for axu in (ax0, ax1, ax2) + ((ax_aux,) if has_aux else ())]

    # (axes, its animated artists), fixed order — the background cache is
    # index-aligned with this list. ax_raw is handled separately.
    _groups = [(ax0, [_markers[0]]), (ax1, [_markers[1]]), (ax2, [_markers[2]])]
    if has_aux:
        _groups.append((ax_aux, [_markers[3]]))
    _state = {'t': None, 'bg': None, 'bg_fig': None, 'bg_chrome': None,
              'raw_region': None, 'chrome_dirty': True,
              'yi': RAW_YLIM_I0, 'wi': RAW_WINDOW_I0, 'raw_anim': False}

    def _half():
        """Current window half-width in seconds."""
        return RAW_WINDOWS[_state['wi']] / 2.0

    # Agg/Qt/Tk all support blitting; a backend that does not needs the markers
    # drawn the ordinary way, so un-animate them and fall back to draw_idle().
    _can_blit = bool(getattr(fig.canvas, 'supports_blit', False))
    if not _can_blit:
        for _m in _markers:
            _m.set_animated(False)

    def _raw_region():
        """Blit region for ax_raw: its tight bbox, padded on all sides.

        Padding is load-bearing, not cosmetic. The region is captured once per
        full draw but must cover every ylim ↑ / ↓ can select afterwards:
          · horizontally, because a wider y tick label ('-1000' vs '-75')
            reaches further left than the one present at capture time;
          · vertically, because the topmost tick label straddles the axes edge,
            so at a scale whose last tick sits exactly at the top (±300 → '300')
            it extends above a tight bbox taken at a scale where it does not.
        Anything drawn outside the region is never restored, and lingers as a
        ghost label. The pad stays well inside the gap to panel 2's x-label.
        """
        bb = ax_raw.get_tightbbox()
        return Bbox.from_extents(max(0.0, bb.x0 - 55.0), max(0.0, bb.y0 - 12.0),
                                 min(fig.bbox.x1, bb.x1 + 4.0),
                                 min(fig.bbox.y1, bb.y1 + 12.0))

    def _capture_bg(_event=None):
        # Fires after every full draw (initial, resize, pan/zoom), so the cache
        # follows the layout. Animated artists are skipped by the renderer,
        # which is exactly what keeps the captured regions clean.
        _state['bg'] = [fig.canvas.copy_from_bbox(a.bbox) for a, _ in _groups]
        if _state['raw_anim']:
            # Region is cached alongside the pixels: restore_region needs the
            # exact bbox the capture was taken with. ax_raw is animated, so
            # these pixels are bare figure background.
            _state['raw_region'] = _raw_region()
            _state['bg_fig'] = fig.canvas.copy_from_bbox(_state['raw_region'])
            _state['chrome_dirty'] = True

    def _rebuild_chrome():
        """Redraw ax_raw's axis furniture and snapshot it for later scrolls."""
        fig.canvas.restore_region(_state['bg_fig'])
        fig.draw_artist(ax_raw)          # patch + grid + ticks; trace is animated
        _state['bg_chrome'] = fig.canvas.copy_from_bbox(_state['raw_region'])
        _state['chrome_dirty'] = False

    def _blit():
        if not _can_blit:
            fig.canvas.draw_idle()
            return
        if _state['bg'] is None:
            fig.canvas.draw()            # triggers _capture_bg
            if _state['bg'] is None:     # no draw_event → nothing to blit onto
                return
        for bg, (a, artists) in zip(_state['bg'], _groups):
            fig.canvas.restore_region(bg)
            for art in artists:
                a.draw_artist(art)
            fig.canvas.blit(a.bbox)
        if _state['bg_fig'] is not None:
            if _state['chrome_dirty']:
                _rebuild_chrome()        # after a zoom or a relayout
            else:
                fig.canvas.restore_region(_state['bg_chrome'])
            ax_raw.draw_artist(raw_line)
            ax_raw.draw_artist(raw_stamp)
            fig.canvas.blit(_state['raw_region'])

    def _draw_raw(t_centre_s):
        """Move the raw window to `t_centre_s` (seconds into the recording).

        The centre is snapped to a whole second so the one-second ruling lines
        up with whole seconds of the recording, not with an arbitrary click
        offset. ← / → keep that alignment (their step is a whole number of
        seconds).
        """
        t_centre_s = float(np.clip(round(t_centre_s), 0.0, sig_dur_s))
        half = _half()
        i0 = max(0, int(round((t_centre_s - half) * fs_128)))
        i1 = min(len(sig_128), int(round((t_centre_s + half) * fs_128)))
        if i1 - i0 < 2:
            return
        _state['t'] = t_centre_s
        raw_line.set_data(np.arange(i0, i1) / fs_128 - t_centre_s,
                          sig_128[i0:i1] * uv_per_lsb)
        _h, _rem = divmod(t_centre_s, 3600.0)
        raw_stamp.set_text(f'{int(_h):02d}:{int(_rem // 60):02d}:'
                           f'{_rem % 60:05.2f}   (t = {t_centre_s / 3600:.3f} h)'
                           f'   {RAW_WINDOWS[_state["wi"]]:g} s'
                           f'  ·  ±{RAW_YLIMS[_state["yi"]]:g} µV')
        for m in _markers:
            m.set_xdata([t_centre_s / 3600.0] * 2)
            m.set_visible(True)
        if not _state['raw_anim'] and _can_blit:
            # One-off upgrade: drop the hint, take ax_raw out of the figure
            # background and the trace out of ax_raw's own draw. Costs one full
            # draw here; every later step, scroll or zoom alike, is blit-only.
            raw_hint.set_visible(False)
            ax_raw.set_animated(True)
            raw_line.set_animated(True)
            raw_stamp.set_animated(True)
            _state['raw_anim'] = True
            _state['bg'] = None
        elif raw_hint.get_visible():
            raw_hint.set_visible(False)  # no-blit backend: plain redraw path
        _blit()

    def _rescale(delta_i):
        """Step the amplitude scale: ↑ magnifies (smaller ylim), ↓ shrinks."""
        i = int(np.clip(_state['yi'] + delta_i, 0, len(RAW_YLIMS) - 1))
        if i == _state['yi']:
            return
        _state['yi'] = i
        ax_raw.set_ylim(-RAW_YLIMS[i], RAW_YLIMS[i])
        _state['chrome_dirty'] = True     # grid + y tick labels moved
        _draw_raw(_half() if _state['t'] is None else _state['t'])

    def _resize(delta_i):
        """Step the window width: + widens, − narrows. Centre time is kept."""
        i = int(np.clip(_state['wi'] + delta_i, 0, len(RAW_WINDOWS) - 1))
        if i == _state['wi']:
            return
        _state['wi'] = i
        ax_raw.set_xlim(-_half(), _half())
        # x tick labels and the one-second ruling both change with the width.
        _state['chrome_dirty'] = True
        _draw_raw(_half() if _state['t'] is None else _state['t'])

    def _on_click(event):
        if event.inaxes not in (ax0, ax1, ax2) + _aux_axes or event.xdata is None:
            return
        _draw_raw(float(event.xdata) * 3600.0)   # x is hours on all three

    def _on_key(event):
        if event.key in ('up', 'down'):
            _rescale(-1 if event.key == 'up' else 1)
            return
        # '=' is the unshifted key that carries '+' on most layouts — accepting
        # it saves the user a shift press.
        if event.key in ('+', '=', '-'):
            _resize(1 if event.key in ('+', '=') else -1)
            return
        step = {'left': -RAW_STEP_S, 'right': RAW_STEP_S,
                'shift+left': -2 * _half(),
                'shift+right': 2 * _half()}.get(event.key)
        if step is None:
            return
        # Before the first click, ← / → open at the start of the recording.
        _draw_raw(_half() if _state['t'] is None else _state['t'] + step)

    fig.canvas.mpl_connect('draw_event', _capture_bg)
    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    # ── Panel 4: sleep statistics table ──────────────────────────────────────
    stats = compute_sleep_stats(preds, epoch_s=30)
    _add_stats_panel(fig, gs[ROW_STATS, 0], stats)

    title = f'Sleep Staging  —  {n_epochs} epochs  ({duration_h:.1f} h)'
    if ch_label:
        title += f'  —  {ch_label}'
    fig.suptitle(title, color=TEXT, fontsize=13, fontweight='700', y=0.975)
    fig.subplots_adjust(top=0.945, bottom=0.04, left=0.05, right=0.965)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Sleep Staging with AnySleep')
    parser.add_argument('file', help='EDF file path')
    parser.add_argument('--channel', '-c', type=int, default=0,
                        help='EEG channel index (0–7), default 0')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to anysleep-run1.pth (auto-download if '
                             'omitted) or to a .onnx file (use with '
                             '--model-type onnx)')
    parser.add_argument('--model-type', choices=['pth', 'onnx'],
                        default='pth',
                        help="Which inference path to take: 'pth' = PyTorch "
                             "reference (default), 'onnx' = the same ONNX the "
                             "phone bundles, run through onnxruntime. Use "
                             "'onnx' to verify that the on-device bytes "
                             "produce the same hypnogram as the PyTorch run.")
    parser.add_argument('--start', '-s', type=float, default=0.0,
                        help='Start time in seconds')
    parser.add_argument('--duration', '-d', type=float, default=-1,
                        help='Duration in seconds (-1 = full file)')
    parser.add_argument('--fmax', type=float, default=30.0,
                        help='Spectrogram max frequency in Hz (default 30, max 60)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to PNG instead of showing interactively')
    args = parser.parse_args()

    # ── Load signal ───────────────────────────────────────────────────────────
    print(f'Loading {args.file}, channel {args.channel}...')
    signal, fs, uv_per_lsb = read_edf_channel(args.file, args.channel)
    print(f'✓ {len(signal)} samples @ {fs:.1f} Hz  ({len(signal)/fs/3600:.2f} h)'
          f'  ·  {uv_per_lsb:.5f} µV/LSB')

    start_idx = int(args.start * fs)
    end_idx   = (start_idx + int(args.duration * fs)) if args.duration > 0 else len(signal)
    signal    = signal[start_idx:end_idx]

    # Battery / light, cut to the same span and re-zeroed like the EEG.
    t0_s = start_idx / fs
    t1_s = end_idx / fs
    aux = {}
    for name, (t_s, v) in read_edf_aux(args.file).items():
        m = (t_s >= t0_s) & (t_s < t1_s)
        if np.any(m):
            aux[name] = (t_s[m] - t0_s, v[m])
    print(f'  Using {len(signal)/fs:.0f} s  ({len(signal)/fs/3600:.2f} h)')

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model is None:
        if args.model_type == 'onnx':
            args.model = Path(__file__).parent / 'anysleep-run1.onnx'
        else:
            args.model = Path(__file__).parent / 'anysleep-run1.pth'
    if args.model_type == 'pth':
        model_path = ensure_model(args.model)
    else:
        # The ONNX bytes are produced by `export_anysleep_onnx.py`; no
        # download flow yet, so just demand the file exists.
        model_path = str(args.model)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f'ONNX model not found: {model_path}\n'
                f'Generate it with `python export_anysleep_onnx.py`.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  ·  model_type={args.model_type}  ·  '
          f'model={model_path}')

    # ── Inference ─────────────────────────────────────────────────────────────
    print('Running inference (1–3 min on CPU)...')
    preds, probs, n_epochs, sig_128 = run_inference(
        signal, fs, model_path, device=device, model_type=args.model_type)
    print(f'✓ {n_epochs} epochs classified')

    stage_counts = {name: int(np.sum(preds == i)) for i, name in enumerate(STAGE_NAMES)}
    print('  ' + '  '.join(f'{k}: {v}' for k, v in stage_counts.items()))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fmax = float(np.clip(args.fmax, 5.0, 60.0))
    ch_label = f'Channel {args.channel} (EEG Ch{args.channel})'
    fig = plot_sleep_staging(preds, probs, n_epochs, sig_128,
                             ch_label=ch_label, fmax=fmax,
                             uv_per_lsb=uv_per_lsb, fs_128=128.0, aux=aux)
    if args.save:
        fig.savefig(args.save, dpi=110, facecolor=DARK, bbox_inches='tight')
        print(f'✓ Plot saved to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
