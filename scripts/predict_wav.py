import argparse
from pathlib import Path

import torch

from AI_dio.data_preprocessing.audio_utils import load_audio_mono_resampled
from AI_dio.data_preprocessing.features import (
    FeatureParams,
    build_mel_transforms,
    mel_tokens_from_audio,
)
from AI_dio.training.checkpoints import load_checkpoint
from AI_dio.training.common import choose_device
from AI_dio.training.models import BaselineCNN

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    wav_path: Path,
    device: torch.device,
    params: FeatureParams,
    threshold: float,
) -> dict:
    target_length = int(params.chunk_duration * params.target_sr)
    mel, to_db = build_mel_transforms(params)
    audio = load_audio_mono_resampled(
        wav_path, target_sr=params.target_sr, target_length=target_length
    )
    tokens = mel_tokens_from_audio(audio, params, mel=mel, to_db=to_db)
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    x = tokens.unsqueeze(0).to(device)
    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    prob_real = float(probs[0].item())
    prob_fake = float(probs[1].item())
    pred = 1 if prob_fake >= threshold else 0
    return {
        "prediction": pred,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "threshold": threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict AI-generated audio from a single .wav file."
    )
    parser.add_argument("wav", type=Path, help="Path to the .wav file.")
    parser.add_argument(
        "--checkpoint", default="checkpoints/model_best.pt", help="Checkpoint path."
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    wav_path = args.wav
    if not wav_path.is_absolute():
        wav_path = (ROOT / wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing wav file: {wav_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = choose_device(args.device)
    model = BaselineCNN()
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    params = FeatureParams()
    out = predict(model, wav_path, device, params, args.threshold)
    label = "fake" if out["prediction"] == 1 else "real"
    print(
        "prediction={label} prob_fake={prob_fake:.4f} prob_real={prob_real:.4f} "
        "threshold={threshold:.2f}".format(
            label=label,
            prob_fake=out["prob_fake"],
            prob_real=out["prob_real"],
            threshold=out["threshold"],
        )
    )
