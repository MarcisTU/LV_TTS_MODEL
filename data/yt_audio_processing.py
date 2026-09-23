import json
import string
import re
from pathlib import Path
import shutil
import os
import subprocess
from pathlib import Path
from yt_dlp import YoutubeDL
from tqdm import tqdm
from loguru import logger


def format_time(seconds: float) -> str:
    """Formats seconds to HH:MM:SS.mmm for ffmpeg."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def parse_sec(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(parts[0]) * 60 + float(parts[1])


def parse_vtt_with_word_timestamps(vtt_path: Path):
    """
    Parses YouTube auto-generated VTT cues and extracts precise word-level
    timestamps while filtering out rolling buffer duplicates.
    """
    if not vtt_path.exists():
        return []

    cue_header_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )
    word_tag_re = re.compile(
        r"<(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})><c>\s*(.*?)\s*</c>"
    )

    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = re.split(r"\n\s*\n", content)
    word_tokens = []

    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        header_match = None
        text_lines = []

        for line in lines:
            m = cue_header_re.search(line)
            if m:
                header_match = m
            elif not line.startswith("WEBVTT") and not line.startswith("NOTE"):
                text_lines.append(line)

        if not header_match or not text_lines:
            continue

        cue_start = parse_sec(header_match.group(1))
        cue_end = parse_sec(header_match.group(2))

        # Ignore 0.01s duplicate snapshot cues YouTube outputs
        if (cue_end - cue_start) <= 0.02:
            continue

        for line in text_lines:
            # Extract first word before any inline <timestamp> tag
            base_word = re.sub(r"<[^>]+>", "", line.split("<")[0]).strip()
            if base_word:
                word_tokens.append({"start": cue_start, "word": base_word})

            # Extract remaining inline words with timestamps
            for w_match in word_tag_re.finditer(line):
                w_time = parse_sec(w_match.group(1))
                w_text = w_match.group(2).strip()
                if w_text:
                    word_tokens.append({"start": w_time, "word": w_text})

    if not word_tokens:
        return []

    # Sort tokens chronologically
    word_tokens.sort(key=lambda x: x["start"])

    # Remove identical consecutive word duplicates within 1.0s window
    deduped_tokens = []
    for t in word_tokens:
        if not deduped_tokens:
            deduped_tokens.append(t)
            continue
        prev = deduped_tokens[-1]
        if t["word"] == prev["word"] and abs(t["start"] - prev["start"]) < 1.0:
            continue
        deduped_tokens.append(t)

    # Calculate realistic end times based on the next word's start time
    words_with_end = []
    for i in range(len(deduped_tokens)):
        w_start = deduped_tokens[i]["start"]
        if i < len(deduped_tokens) - 1:
            next_start = deduped_tokens[i + 1]["start"]
            # A word typically lasts between 0.25s and 1.2s
            w_end = min(next_start, w_start + 1.2)
        else:
            w_end = w_start + 0.8

        words_with_end.append(
            {"start": w_start, "end": w_end, "word": deduped_tokens[i]["word"]}
        )

    # Filter youtube repeating segments
    words_with_end_processed = []
    for wwe in words_with_end:
        if wwe["end"] - wwe["start"] > 0.05:
            words_with_end_processed.append(wwe) 

    return words_with_end_processed


def assemble_phrases_from_words(words, max_gap=1.2, max_phrase_duration=10.0):
    """
    Groups words into short phrases based on silence gaps and duration limits.
    """
    if not words:
        return []

    phrases = []
    curr_words = [words[0]["word"]]
    curr_start = words[0]["start"]
    curr_end = words[0]["end"]

    for w in words[1:]:
        gap = w["start"] - curr_end
        current_dur = w["end"] - curr_start

        # Break into a new phrase if pause > max_gap OR phrase length > max_phrase_duration
        if gap > max_gap or current_dur > max_phrase_duration:
            phrases.append(
                {
                    "start": round(curr_start, 3),
                    "end": round(curr_end, 3),
                    "text": " ".join(curr_words).strip(),
                }
            )
            curr_words = [w["word"]]
            curr_start = w["start"]
            curr_end = w["end"]
        else:
            curr_words.append(w["word"])
            curr_end = max(curr_end, w["end"])

    if curr_words:
        phrases.append(
            {
                "start": round(curr_start, 3),
                "end": round(curr_end, 3),
                "text": " ".join(curr_words).strip(),
            }
        )

    # text processing and cleanup
    for phrase in phrases:
        phrase_text = phrase["text"]
        phrase_text = phrase_text.replace("&gt;&gt;", "").capitalize()
        phrase_text = re.sub(r" +", " ", phrase_text)

        if not phrase_text.endswith(tuple(string.punctuation)):
            phrase_text = f"{phrase_text}."

        phrase["text"] = phrase_text

    return phrases


def process_youtube_url(url, output_dir="output_dataset", target_lang="lv"):
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path / "train_raw.jsonl"
    temp_dir = output_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--> Extracting: {url}")

    ydl_opts = {
        "format": "bestaudio/best",
        "writepath": str(temp_dir),
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [target_lang],
        "subtitlesformat": "vtt",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    full_audio_file = temp_dir / f"{video_id}.wav"
    vtt_candidates = list(temp_dir.glob(f"{video_id}*.{target_lang}.vtt"))

    if not vtt_candidates:
        logger.info(f"[-] No '{target_lang}' subtitles found.")
        return

    vtt_file = vtt_candidates[0]

    # 1. Parse word tokens with accurate word-level end times
    words = parse_vtt_with_word_timestamps(vtt_file)

    # 2. Assemble small phrases with hard duration checks
    final_segments = assemble_phrases_from_words(
        words, max_gap=1.2, max_phrase_duration=10.0
    )

    if not final_segments:
        print("[-] No valid segments found.")
        return

    existing_count = 0
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())

    entries = []
    logger.info(f"--> Exporting {len(final_segments)} optimized segments (target <= 12s)...")

    for idx, seg in enumerate(final_segments, start=existing_count + 1):
        utt_id = f"utt{idx:04d}"
        output_wav_relative = f"./audio/{utt_id}.wav"
        output_wav_full = audio_dir / f"{utt_id}.wav"
        duration = seg["end"] - seg["start"]
        start_str = format_time(seg["start"])

        # FFmpeg trimming with fast input-seeking
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", start_str,
            "-i", str(full_audio_file),
            "-t", f"{duration:.3f}",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1", str(output_wav_full),
            "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True)

        entries.append(
            {
                "audio": output_wav_relative,
                "text": seg["text"],
                "language": target_lang,
            }
        )

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"[+] Done! Appended {len(entries)} optimized segments to {jsonl_path}\n")
    logger.info(f"Total count of samples: {existing_count + len(entries)}")

    os.remove(full_audio_file)
    os.remove(vtt_file)


if __name__ == "__main__":
    target_dir = Path("./output_dataset")
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cleaning target directory...")
    for item in target_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    urls = [
        "https://www.youtube.com/watch?v=MnnzEDxEdbI",
        "https://www.youtube.com/watch?v=aQFNfYMgSGY",
        "https://www.youtube.com/watch?v=JRSfeoogsJo",
        "https://www.youtube.com/watch?v=QvUuYIWd4jA",
        "https://www.youtube.com/watch?v=KskWI0-b5P8",
    ]
    for url in tqdm(urls, desc="Processing videos..."):
        process_youtube_url(url, output_dir=target_dir)
