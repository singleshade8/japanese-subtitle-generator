import streamlit as st
import subprocess
import os
import time
import torch
from faster_whisper import WhisperModel

# ================= CONFIG =================
FFMPEG_PATH = r"C:\projects\SubtitleGenerator\ffmpeg\bin\bin\ffmpeg.exe"
MODEL_SIZE = "small"   # change to "medium" if needed
# =========================================

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
COMPUTE_TYPE = "float16" if USE_GPU else "int8"

def format_eta(seconds):
    seconds = int(seconds)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}m {secs}s"

def format_ts(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# ---------------- UI ----------------
st.set_page_config(page_title="Japanese → English Subtitle Generator")

st.title("🎬 Japanese → English Subtitle Generator")
st.write("GPU accelerated • No size limit • Uncensored")

st.info("🚀 GPU (faster-whisper)" if USE_GPU else "🐢 CPU (slower)")

video_path = st.text_input(
    "Enter full video path",
    placeholder=r"C:\Users\ASUS\Downloads\Telegram Desktop\The_Big_Bang_Theory_S06E14.mkv"
)

# ---------- Preview Panel ----------
st.subheader("📺 Subtitle Preview (live)")
preview_box = st.empty()

# ---------- Progress ----------
progress_bar = st.progress(0)
status_text = st.empty()
eta_text = st.empty()

if st.button("🚀 Generate Subtitles"):
    if not video_path or not os.path.exists(video_path):
        st.error("Invalid video path")
        st.stop()

    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    srt_filename = base_name + "_en.srt"
    srt_path = os.path.join(os.path.dirname(video_path), srt_filename)

    audio_path = "audio.wav"

    # ---- Extract audio ----
    status_text.text("🎧 Extracting audio...")
    subprocess.run([
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    progress_bar.progress(10)

    # ---- Load faster-whisper ----
    status_text.text("🧠 Loading faster-whisper model...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    progress_bar.progress(20)

    # ---- Transcribe (single pass, no chunking) ----
    status_text.text("🌍 Translating Japanese → English...")
    start_time = time.time()

    segments, info = model.transcribe(
        audio_path,
        language="ja",
        task="translate",
        beam_size=5
    )

    total_segments = 0
    collected = []

    # Count segments first (for ETA)
    for _ in segments:
        total_segments += 1

    # Re-run generator
    segments, _ = model.transcribe(
        audio_path,
        language="ja",
        task="translate",
        beam_size=5
    )

    for i, seg in enumerate(segments, start=1):
        collected.append(seg)

        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (total_segments - i)

        progress = 20 + int((i / total_segments) * 70)
        progress_bar.progress(progress)

        status_text.text(f"📝 Processing subtitle {i}/{total_segments}")
        eta_text.text(f"⏱ ETA: {format_eta(remaining)}")

        # Live preview
        preview_box.markdown(
            f"**[{format_ts(seg.start)} → {format_ts(seg.end)}]**  \n{seg.text}"
        )

    # ---- Write SRT ----
    status_text.text("📝 Writing SRT file...")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(collected, 1):
            f.write(f"{i}\n")
            f.write(
                f"{format_ts(seg.start)} --> {format_ts(seg.end)}\n"
            )
            f.write(seg.text.strip() + "\n\n")

    progress_bar.progress(100)
    status_text.text("✅ Done!")
    eta_text.text("")

    st.success(f"Subtitles created: {srt_filename}")