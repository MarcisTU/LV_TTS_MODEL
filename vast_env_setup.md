


conda packages:
```bash
conda create -n lv_tts python=3.12
conda activate lv_tts

# code and packages
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune]"
pip install loguru
```


# local wsl project path data
-> \\wsl.localhost\Ubuntu\home\marcis\moss_tts_lv_model\data\


## process codes on server (edit dataset_root)
```bash
python ./MOSS-TTS/moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --dataset_root /workspace/LV_TTS_MODEL/data/output_dataset \
    --input-jsonl ./data/output_dataset/train_raw.jsonl \
    --output-jsonl ./data/output_dataset/train_with_codes.jsonl
```

# train
```bash
accelerate launch ./MOSS-TTS/moss_tts_local/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --train-jsonl ./data/output_dataset/train_with_codes.jsonl \
    --output-dir output/moss_tts_local_sft \
    --per-device-batch-size 6 \
    --gradient-accumulation-steps 2 \
    --learning-rate 4e-5 \
    --warmup-steps 50 \
    --num-epochs 20 \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```
