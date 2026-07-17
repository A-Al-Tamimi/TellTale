# TellTale

Text-only ambivalence/hesitancy (A/H) recognition for the BAH dataset.
Submitted entry to the **3rd A/H Video Recognition Challenge** (11th ABAW
Workshop, ECCV 2026): private-test **Macro-F1 0.7364**, AP 0.7940, versus
the official baseline of 0.2827.

## Method in brief

The name says it: hesitancy leaves telltale signs in what a speaker
*tells* you. TellTale ignores the video and audio entirely and classifies
the interview transcript with three blended probability streams:

1. **Stream A** — multilingual-e5-large, fine-tuned with LoRA adapters.
   Trained in two phases: first on individual transcript chunks (chunk
   labels derived from the dataset's annotated A/H time segments), then
   under a video-level multiple-instance (MIL) objective.
2. **Stream B** — mDeBERTa-v3-base, fine-tuned with LoRA directly under
   the same MIL objective. Weaker alone, but its errors differ from
   stream A's, which the blend exploits.
3. **Stream J** — a zero-shot judge: Qwen3-14B (4-bit) is prompted to rate
   each transcript 0–100 for A/H. No training, no labeled examples.The
   model runs locally through [`mlx-lm`](https://github.com/ml-explore/mlx-lm)

The MIL objective pools each video's per-chunk scores with a smooth
maximum (LogSumExp, r=50), so the single most hesitant-sounding chunk
determines the video score and only the video-level label is needed.
Each encoder stream is an average over per-fold, per-seed checkpoints
(stream A: 5 folds x 3 seeds; stream B: 5 folds x 2 seeds).

Final prediction: `p = 0.45*pA + 0.30*pB + 0.25*pJ`, positive if
`p >= 0.53`. The weights and threshold were selected on participant-grouped
5-fold out-of-fold predictions only; the test set played no part in any
selection decision.

## Requirements and assumptions

- **Data access.** The BAH dataset is distributed under an EULA by the
  challenge organizers; this repository contains no data. You need:
  - the labeled release (directory with `split/{train,val,test}.txt`,
    `transcription/`, and `video_annotation_transcript.yaml`), and
  - the test release (directory with `split/test.txt`, `transcription/`,
    and `video.csv` giving the required submission row order).
- **Hardware.** Streams A and B train with PyTorch on Apple Silicon
  (`--device mps`, default), CUDA (`--device cuda`), or CPU. Expect
  roughly 0.5–1 h per fold-seed for stream A and less for stream B; 25
  fold-seed runs in total. Stream J requires **Apple Silicon** (MLX,
  ~9 GB for the 4-bit model, ~5 s per video); on other hardware you can
  substitute any runtime that produces the same 0–100 JSON ratings and
  write them to `outputs/judge_*.jsonl` in the same format.
- **Python** 3.10+.

## Setup

```bash
pip install -r requirements.txt
# Apple Silicon only, needed for stream J:
pip install mlx-lm==0.28.0

export BAH_DIR=/path/to/BAH/data          # labeled release
export BAH_TEST_DIR=/path/to/BAH_test/data  # test release
export TELLTALE_OUT=outputs               # optional; default ./outputs
```

## Reproducing the submission

```bash
# 1. Train both encoder streams (all folds and seeds; resumable, ~15-25 h total)
python telltale.py train --stream a
python telltale.py train --stream b

# 2. Run the zero-shot judge over the labeled pool and the test set
python telltale.py judge --split labeled
python telltale.py judge --split test

# 3. Sanity-check out-of-fold scores (expected: A ~0.71, B ~0.67, judge ~0.70,
#    blend ~0.72 Macro-F1)
python telltale.py oof

# 4. Write the submission file (canonical row order from video.csv)
python telltale.py predict --out submission.txt
```

Every step is resumable: finished fold-seed runs and judged videos are
cached under `outputs/` and skipped on re-run. `python telltale.py blend`
re-derives the blend weights and flat-region threshold from your own OOF
predictions if you wish to verify the shipped/suggested constants.

Note on exact reproduction: fold assignment is deterministic
(participant-grouped, no randomness), but per-run training numbers vary
slightly across hardware and library versions. The multi-seed checkpoint
averaging is what makes the final scores stable; individual fold-seed runs
should not be compared in isolation.

## Citation

If you use this code or build on the method, please cite:

```bibtex
@inproceedings{altamimi2026telltale,
  title     = {Telltale: Blending Multi-Instance LoRA Text Encoders and a Zero-Shot
               LLM Judge for Ambivalence/Hesitancy Recognition in Videos},
  author    = {Al-Tamimi, Abdel-Karim},
  booktitle = {Proceedings of the 11th Workshop on Affective Behavior
               Analysis in-the-Wild (ABAW), ECCV},
  year      = {2026}
}
```

## License

The BAH dataset itself is governed by its own EULA and is not included or redistributed here.
