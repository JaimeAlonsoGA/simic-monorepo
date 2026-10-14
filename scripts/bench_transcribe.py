#!/usr/bin/env python
"""Transcription benchmark — compare configurations on speed AND quality.

Tuning this project's Whisper settings has twice been reverted after a change that
felt right measured badly (a shortened temperature ladder, a lowered chunk_length),
so configuration choices are made from a table, not an impression. This harness
produces that table.

It drives simic's own `Transcriber`, not a private copy of the decode call, so
what it measures is what the tool ships.

Usage:

    uv run python scripts/bench_transcribe.py --manifest my-bench.json \\
        --models large-v3-turbo,large-v3 \\
        --compute-types float16,int8_float16 \\
        --hotwords-modes off,on \\
        --beam-sizes 1,4 --cpu-threads 0,8 \\
        --output-dir reports/bench

The manifest is deliberately NOT committed: it points at personal recordings. Copy
`scripts/bench_manifest.example.json` and fill in your own paths and the terms each
recording is expected to contain. An entry with a `reference` transcript is also
scored by word error rate; `scripts/make_bench_meeting.py` builds such a corpus from
public speech when there are no transcribed recordings of one's own.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simic._gpu import sample_gpu, wait_for_clamp_release
from simic._quality import (
    count_hallucination_markers,
    count_recognised_terms,
    count_repeated_phrases,
    count_repeated_words,
    find_hallucination_markers,
    low_confidence_rate,
    punctuation_per_1000_words,
    word_error_rate,
)
from simic.formatter import WORD_LOW_CONFIDENCE
from simic.settings import Settings
from simic.transcriber import Transcriber


@dataclass
class Result:
    """One grid point: a configuration measured against one recording."""

    recording: str
    model: str
    compute_type: str
    hotwords: str
    beam_size: int
    cpu_threads: int
    batch_size: int
    duration_seconds: float
    elapsed_seconds: float
    rtf: float
    segments: int
    characters: int
    hallucination_markers: int
    hallucinations_found: list[str]
    repeated_words: int
    repeated_phrases: int
    punctuation_per_1000: float
    low_confidence_percent: float
    terms_expected: int
    terms_found: int
    terms_missing: list[str]
    # Against the manifest's reference transcript; None when the entry has none.
    wer_percent: float | None = None
    # The one-line GPU summary as the pipeline reports it (clocks, temperature,
    # throttle share) — kept as text so the benchmark and the tool say the same thing.
    gpu: str | None = None
    error: str | None = None
    events: list[str] = field(default_factory=list)


def _load_manifest(path: Path) -> list[dict]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError("manifest must be a JSON list of {path, terms} objects")
    for entry in entries:
        if "path" not in entry:
            raise ValueError(f"manifest entry missing 'path': {entry}")
    return entries


def run_one(
    audio: Path,
    entry: dict,
    settings: Settings,
    *,
    hotwords_mode: str,
) -> Result:
    """Measure one configuration against one recording."""
    terms: list[str] = entry.get("terms", [])
    reference: str | None = entry.get("reference")

    captured: list[str] = []
    gpu_line: list[str] = []

    transcriber = Transcriber(settings)
    started = time.monotonic()
    try:
        with sample_gpu(gpu_line.append, enabled=settings.device == "cuda"):
            segments, info = transcriber.transcribe(
                audio, stage=audio.stem, on_status=captured.append
            )
    except Exception as exc:
        return Result(
            recording=audio.name,
            model=settings.stt_model,
            compute_type=settings.compute_type,
            hotwords=hotwords_mode,
            beam_size=settings.beam_size,
            cpu_threads=settings.cpu_threads,
            batch_size=settings.batch_size,
            duration_seconds=0.0,
            elapsed_seconds=time.monotonic() - started,
            rtf=0.0,
            segments=0,
            characters=0,
            hallucination_markers=0,
            hallucinations_found=[],
            repeated_words=0,
            repeated_phrases=0,
            punctuation_per_1000=0.0,
            low_confidence_percent=0.0,
            terms_expected=len(terms),
            terms_found=0,
            terms_missing=list(terms),
            error=f"{type(exc).__name__}: {exc}",
            events=captured,
        )
    finally:
        del transcriber

    elapsed = time.monotonic() - started
    text = " ".join(segment.text for segment in segments)
    duration = float(info.get("duration", 0.0))
    found, missing = count_recognised_terms(text, terms)
    probabilities = [word.probability for segment in segments for word in (segment.words or [])]

    return Result(
        recording=audio.name,
        model=settings.stt_model,
        compute_type=settings.compute_type,
        hotwords=hotwords_mode,
        beam_size=settings.beam_size,
        cpu_threads=settings.cpu_threads,
        batch_size=settings.batch_size,
        duration_seconds=duration,
        elapsed_seconds=round(elapsed, 1),
        rtf=round(duration / elapsed, 2) if elapsed else 0.0,
        segments=len(segments),
        characters=len(text),
        hallucination_markers=count_hallucination_markers(text),
        hallucinations_found=find_hallucination_markers(text),
        repeated_words=count_repeated_words(text),
        repeated_phrases=count_repeated_phrases(text),
        punctuation_per_1000=round(punctuation_per_1000_words(text), 1),
        low_confidence_percent=round(low_confidence_rate(probabilities, WORD_LOW_CONFIDENCE), 1),
        terms_expected=len(terms),
        terms_found=found,
        terms_missing=missing,
        wer_percent=None if reference is None else round(word_error_rate(reference, text), 1),
        gpu=gpu_line[0] if gpu_line else None,
        events=captured,
    )


def _config_label(r: Result) -> str:
    return (
        f"{r.model} | {r.compute_type} | {r.hotwords} | beam {r.beam_size} | "
        f"threads {r.cpu_threads or 'default'} | batch {r.batch_size}"
    )


_CONFIG_HEADER = "model | compute | hotwords | beam | threads | batch"
_CONFIG_RULE = "---|---|---|---|---|---"


def _wer(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _markdown_table(results: list[Result]) -> str:
    header = (
        f"| recording | {_CONFIG_HEADER} | elapsed | RTF | WER % | segs | "
        "hallu | rep-w | rep-p | terms |\n"
        f"|---|{_CONFIG_RULE}|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r.recording} | {_config_label(r)} | "
        f"{r.elapsed_seconds:.1f}s | {r.rtf:.2f}x | {_wer(r.wer_percent)} | {r.segments} | "
        f"{r.hallucination_markers} | {r.repeated_words} | {r.repeated_phrases} | "
        f"{r.terms_found}/{r.terms_expected} |"
        for r in results
    ]
    return "\n".join([header, *rows])


def _summary(results: list[Result]) -> str:
    """Aggregate per configuration, so the winner is visible without reading rows."""
    by_config: dict[str, list[Result]] = {}
    for r in results:
        by_config.setdefault(_config_label(r), []).append(r)

    lines = [
        f"| {_CONFIG_HEADER} | total time | median RTF | WER % | hallu | loops | "
        "low-conf % | punct/1k | terms |",
        f"|{_CONFIG_RULE}|---|---|---|---|---|---|---|---|",
    ]
    for label, group in sorted(by_config.items()):
        found = sum(r.terms_found for r in group)
        expected = sum(r.terms_expected for r in group)
        loops = sum(r.repeated_words + r.repeated_phrases for r in group)
        scored = [r.wer_percent for r in group if r.wer_percent is not None]
        lines.append(
            f"| {label} | {sum(r.elapsed_seconds for r in group):.0f}s | "
            f"{statistics.median(r.rtf for r in group):.2f}x | "
            f"{_wer(statistics.mean(scored) if scored else None)} | "
            f"{sum(r.hallucination_markers for r in group)} | {loops} | "
            f"{statistics.median(r.low_confidence_percent for r in group):.1f} | "
            f"{statistics.median(r.punctuation_per_1000 for r in group):.0f} | "
            f"{found}/{expected} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", default="large-v3-turbo")
    parser.add_argument("--compute-types", default="float16")
    parser.add_argument("--hotwords-modes", default="off")
    parser.add_argument("--hotwords", default="", help="glossary used when mode is 'on'")
    parser.add_argument("--beam-sizes", default="4")
    parser.add_argument("--cpu-threads", default="0", help="0 is faster-whisper's default (4)")
    parser.add_argument("--batch-sizes", default="0", help="0 disables batched inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/bench"))
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help=(
            "Idle gap between grid points. On a thermally constrained laptop a long "
            "grid heat-soaks the chassis, so later configurations run clamped and the "
            "table measures queue position as much as configuration. Set 60-120 there."
        ),
    )
    parser.add_argument(
        "--clamp-timeout",
        type=float,
        default=0.0,
        help=(
            "Before each grid point, wait up to this many seconds for a GPU thermal "
            "clamp to release. A clamped card timed the same configuration at 3519s "
            "against 41s unclamped, so without this a long grid tabulates thermal "
            "state instead of settings. 0 disables the wait."
        ),
    )
    args = parser.parse_args()

    entries = _load_manifest(args.manifest)
    grid = list(
        itertools.product(
            args.models.split(","),
            args.compute_types.split(","),
            args.hotwords_modes.split(","),
            [int(v) for v in args.beam_sizes.split(",")],
            [int(v) for v in args.cpu_threads.split(",")],
            [int(v) for v in args.batch_sizes.split(",")],
            entries,
        )
    )

    results: list[Result] = []
    for index, (model, compute, mode, beam, threads, batch, entry) in enumerate(grid, start=1):
        audio = Path(entry["path"]).expanduser()
        print(
            f"[{index}/{len(grid)}] {model}/{compute} hotwords={mode} beam={beam} "
            f"threads={threads} batch={batch} {audio.name}",
            flush=True,
        )
        if not audio.exists():
            print(f"  skipped: {audio} not found", flush=True)
            continue
        if args.clamp_timeout:
            wait_for_clamp_release(
                args.clamp_timeout,
                report=lambda m: print(f"  {m}", flush=True),
            )
        settings = Settings(
            stt_model=model,
            compute_type=compute,
            device=args.device,
            language=args.language,
            hotwords=args.hotwords if mode == "on" else "",
            beam_size=beam,
            cpu_threads=threads,
            batch_size=batch,
            run_log=False,
            # A grid measures every point from scratch; a cached channel would time 0 s.
            resume_cache=False,
        )
        result = run_one(audio, entry, settings, hotwords_mode=mode)
        results.append(result)
        print(
            f"  {result.elapsed_seconds:.1f}s RTF={result.rtf:.2f}x "
            f"WER={_wer(result.wer_percent)} hallu={result.hallucination_markers} "
            f"terms={result.terms_found}/{result.terms_expected}"
            + (f" ERROR {result.error}" if result.error else ""),
            flush=True,
        )
        if result.gpu:
            print(f"  {result.gpu}", flush=True)
        if args.pause_seconds and index < len(grid):
            time.sleep(args.pause_seconds)

    if not results:
        print("No runs completed — check the manifest paths.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"bench-{stamp}.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
    md_path = args.output_dir / f"bench-{stamp}.md"
    md_path.write_text(
        f"# Transcription benchmark {stamp}\n\n## Per run\n\n{_markdown_table(results)}\n\n"
        f"## Per configuration\n\n{_summary(results)}\n"
    )
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
