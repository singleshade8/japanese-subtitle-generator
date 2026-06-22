import streamlit as st
import subprocess
import os
import time
import torch
from faster_whisper import WhisperModel

# ================= CONFIG =================
FFMPEG_PATH = r"C:\projects\SubtitleGenerator\ffmpeg\bin\bin\ffmpeg.exe"
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

def is_repetitive(text):
    words = text.lower().split()

    if len(words) < 6:
        return False

    unique_ratio = len(set(words)) / len(words)

    if unique_ratio < 0.35:
        return True

    for word in set(words):
        if words.count(word) > len(words) * 0.5:
            return True

    return False

# ---------------- UI ----------------
st.set_page_config(page_title="Japanese → English Subtitle Generator")

st.title("🎬 Japanese → English Subtitle Generator")

mode = st.selectbox("Mode", ["Fast", "Accurate"])
MODEL_SIZE = "small" if mode == "Fast" else "large-v3"

st.write(f"{'GPU' if USE_GPU else 'CPU'} • Model: {MODEL_SIZE} • No size limit")
st.info("🚀 GPU (faster-whisper)" if USE_GPU else "🐢 CPU (slower)")

video_path = st.text_input(
    "Enter full video path",
    placeholder=r"C:\Users\ASUS\Downloads\Telegram Desktop\video.mkv"
)
video_path = video_path.strip().strip('"')
# ---------- Preview Panel ----------
st.subheader("📺 Subtitle Preview (live)")
preview_box = st.empty()

# ---------- Progress ----------
progress_bar = st.progress(0)
status_text = st.empty()
eta_text = st.empty()

if st.button("🚀 Generate Subtitles"):
    if not video_path or not os.path.exists(video_path):
        st.error("❌ Invalid video path")
        st.stop()

    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    srt_filename = base_name + "_en.srt"

    # Save SRT next to video (with fallback)
    output_dir = os.path.dirname(video_path)
    if not os.access(output_dir, os.W_OK):
        output_dir = os.getcwd()
    srt_path = os.path.join(output_dir, srt_filename)

    audio_path = "audio.wav"

    # ---- Extract audio ----
    status_text.text("🎧 Extracting audio...")
    result = subprocess.run(
    [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ],
    capture_output=True,
    text=True
    )

    if result.returncode != 0:
        st.error("❌ FFmpeg failed")
        st.text(result.stderr)
        st.stop()

    if not os.path.exists(audio_path):
        st.error("❌ Audio extraction failed")
        st.stop()
    audio_size = os.path.getsize(audio_path) / (1024 * 1024)
    st.write(f"🎵 Audio extracted: {audio_size:.2f} MB")
    progress_bar.progress(10)

    #----------chunks----------
    chunk_dir = "chunks"
    if os.path.exists(chunk_dir):
        for f in os.listdir(chunk_dir):
            if f.endswith(".wav"):
                os.remove(os.path.join(chunk_dir, f))

    os.makedirs(chunk_dir, exist_ok=True)

    status_text.text("✂️ Splitting audio into chunks...")

    subprocess.run([
        FFMPEG_PATH,
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", "300",  # 5 minutes
        "-c", "copy",
        os.path.join(chunk_dir, "chunk_%03d.wav")
    ], capture_output=True)

    chunk_files = sorted([
        os.path.join(chunk_dir, f)
        for f in os.listdir(chunk_dir)
        if f.endswith(".wav")
    ])

    st.write(f"📦 Chunks created: {len(chunk_files)}")
    # ---- Load faster-whisper ----
    status_text.text("🧠 Loading faster-whisper model...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    progress_bar.progress(20)

    # ---- Transcribe ----
    status_text.text("🌍 Translating Japanese → English...")
    start_time = time.time()

    st.write("🔍 Debug: Before transcribe")
    st.write(f"GPU Available: {torch.cuda.is_available()}")
    st.write(f"Device: {DEVICE}")
    st.write(f"Model: {MODEL_SIZE}")

    try:
        
        collected = []

        for chunk_index, chunk_file in enumerate(chunk_files):

            status_text.text(
                f"🎙 Processing chunk {chunk_index+1}/{len(chunk_files)}"
            )

            offset = chunk_index * 300

            segments, info = model.transcribe(
                chunk_file,
                language="ja",
                task="translate",
                beam_size=1,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
                word_timestamps=False,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            for seg in segments:

                text = seg.text.strip()

                if len(text) < 2:
                    continue

                if is_repetitive(text):
                    continue

                # shift timestamps by chunk offset
                collected.append(
                    (
                        seg.start + offset,
                        seg.end + offset,
                        text
                    )
                )

            progress = 20 + int(
                ((chunk_index + 1) / len(chunk_files)) * 70
            )

            progress_bar.progress(progress)

        st.write(f"✅ Total Segments: {len(collected)}")
        collected = [
            (start, end, text)
            for start, end, text in collected
            if (end - start) < 60
        ]

        collected = [
            (start, end, text)
            for start, end, text in collected
            if not is_repetitive(text)
        ]

        st.write(f"✅ Segments after filtering: {len(collected)}")
        
    except Exception as e:
        st.error(f"❌ Transcription Error: {str(e)}")
        st.stop()
    # 🔧 Clean hallucinations

        # elapsed = time.time() - start_time
        # avg_time = elapsed / i
        # remaining = avg_time * (total_segments - i) if total_segments else 0

        # progress = 20 + int((i / max(total_segments, 1)) * 70)
        # progress_bar.progress(progress)

        # status_text.text(f"📝 Processing subtitle {i}/{total_segments}")
        # eta_text.text(f"⏱ ETA: {format_eta(remaining)}")

        # preview_box.markdown(
        #     f"**[{format_ts(seg.start)} → {format_ts(seg.end)}]**  \n{seg.text}"
        # )

    # ---- Write SRT ----
    status_text.text("📝 Writing SRT file...")

    with open(srt_path, "w", encoding="utf-8") as f:

        subtitle_num = 1

        for start, end, text in collected:

            if len(text) < 2:
                continue

            if is_repetitive(text):
                continue

            f.write(f"{subtitle_num}\n")
            f.write(
                f"{format_ts(start)} --> {format_ts(end)}\n"
            )
            f.write(text + "\n\n")

            subtitle_num += 1

    progress_bar.progress(100)
    status_text.text("✅ Done!")
    eta_text.text("")

    st.success(f"🎉 Subtitles created: {srt_filename}")
    st.info(f"📁 Saved at: {srt_path}")