from pathlib import Path
import importlib.util
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor


# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


# pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-v1.5"
pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = "This place [pause 2.3s] makes me feel very weird."
# text_2 = "We stand on the threshold of the AI era.\nArtificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."

text_7 = "Take me to the other side [pause 2.8s] please."
text_8 = "For this to work [pause 3.2s] we need a new plan."

# Use audio from ./assets/audio to avoid downloading from the cloud.
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Voice cloning (with reference)
    # [processor.build_user_message(text=text_1, reference=[ref_audio_2])],
    # Direct TTS (no reference). Language tags are recommended in v1.5.
    # [processor.build_user_message(text=text_1)],
    [processor.build_user_message(text=text_1)],
    # Direct TTS (no reference). For languages other than Chinese and English,
    # set the language tag whenever it is known.
    [processor.build_user_message(text=text_7, language="English")],

    # Explicit pause control. Use [pause X.Ys], such as [pause 3.2s].
    [processor.build_user_message(text=text_8)],

    # Duration control
    [processor.build_user_message(text=text_1, tokens=128)],
    [processor.build_user_message(text=text_1, tokens=256)],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    dtype=dtype,
).to(device)
model.eval()

batch_size = 1

save_dir = Path("inference_root")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1

            # if audio.ndim == 1:
            #     audio = audio.unsqueeze(0)
            # torchaudio.save(
            #     str(out_path),
            #     audio.detach().cpu().to(torch.float32),
            #     processor.model_config.sampling_rate,
            #     backend="soundfile"
            # )

            # Convert PyTorch tensor to 1D or 2D NumPy array (squeeze unneeded batch dims)
            audio_np = audio.detach().cpu().to(torch.float32).numpy().squeeze()

            # soundfile expects shape (samples,) for mono or (samples, channels) for stereo
            if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.T  # Transpose (channels, samples) -> (samples, channels)

            sf.write(
                file=str(out_path),
                data=audio_np,
                samplerate=processor.model_config.sampling_rate,
            )
