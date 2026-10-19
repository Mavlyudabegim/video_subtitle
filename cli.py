"""Command line usage:

    python cli.py input.mp4                    # -> input_subtitled.mp4 + .srt
    python cli.py input.mp4 -o out.mp4 --model base --words 3 --plain
"""
import argparse
from pathlib import Path

from subtitler import add_subtitles


def main() -> None:
    p = argparse.ArgumentParser(description="Add English subtitles to a short video.")
    p.add_argument("input", help="path to the input video")
    p.add_argument("-o", "--output", help="output .mp4 (default: <input>_subtitled.mp4)")
    p.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--words", type=int, default=4, help="max words per caption")
    p.add_argument("--plain", action="store_true", help="plain SRT style instead of reels style")
    p.add_argument("--lang", default="en", choices=["en", "uz"],
    help="spoken language (uz uses the model in UZ_WHISPER_MODEL)")
   
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(f"{inp.stem}_subtitled.mp4")
    res = add_subtitles(inp, out, model_size=args.model, language=args.lang, max_words=args.words, styled=not args.plain)
    print(f"Video: {res.video}\nSRT:   {res.srt}\nText:  {res.text}")


if __name__ == "__main__":
    main()