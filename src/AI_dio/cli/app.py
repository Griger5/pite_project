import argparse
import logging
from pathlib import Path

from AI_dio.audio.audio_file_reader import (
    compute_log_mel_spectrogram,
    get_sound_parameters,
    plot_melspectrogram,
    plot_waveform,
    read_sound,
)
from AI_dio.audio.microphone_input import microphone_input

logging.getLogger().setLevel(logging.CRITICAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick audio analysis")

    parser.add_argument(
        "-f", "--file", type=str, help="Audio file to analyse, .mp3 and .wav supported"
    )
    parser.add_argument(
        "-wv",
        "--plot_waveform",
        action="store_true",
        help="Generate and save a waveform of given audio",
    )
    parser.add_argument(
        "-wvf",
        "--waveform_file",
        type=str,
        default="waveform.png",
        help="Filename for the waveform. Default: waveform.png",
    )
    parser.add_argument(
        "-sp",
        "--plot_spectogram",
        action="store_true",
        help="Generate and save a Mel spectogram of the given audio",
    )
    parser.add_argument(
        "-spf",
        "--spectogram_file",
        type=str,
        default="spectogram.png",
        help="Filename for the spectogram. Default: spectogram.png",
    )
    parser.add_argument(
        "-m",
        "--microphone",
        action="store_true",
        help="Record sound from the microphone instead of using a file",
    )
    parser.add_argument(
        "-ms",
        "--microphone_seconds",
        type=int,
        default=5,
        help="Specify the numbers of seconds to record when using a microphone. Default: 5",
    )
    parser.add_argument(
        "-msr",
        "--microphone_sample_rate",
        type=int,
        default=44100,
        help="Specify the sample rate when using a microphone. Default: 44100",
    )
    parser.add_argument(
        "-np",
        "--no_paramaters",
        action="store_true",
        help="Skip printing sound parameters",
    )

    args = parser.parse_args()

    if not args.file and not args.microphone:
        print('No action taken. Maybe try the "--help" command?')

    if args.file:
        audio, parameters = read_sound(Path(args.file))
    elif args.microphone:
        audio, rate = microphone_input(
            args.microphone_seconds, rate=args.microphone_sample_rate
        )
        parameters = get_sound_parameters(audio, rate)

    if not args.no_parameters:
        for key, value in parameters.items():
            print(f"{key}: {value}")

    if args.plot_waveform:
        plot_waveform(audio, Path(args.waveform_file))
        print(f"Waveform saved to {args.waveform_file}")

    if args.plot_spectogram:
        spectogram = compute_log_mel_spectrogram(audio, parameters["sample_rate"])
        plot_melspectrogram(
            spectogram,
            parameters["sample_rate"],
            output_path=Path(args.spectogram_file),
        )
        print(f"Spectogram saved to {args.spectogram_file}")
