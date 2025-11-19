# Crime Activity Video Analyzer

This repository contains a lightweight, heuristic-based video analyzer that samples frames from any video placed in the project folder and attempts to classify the clip into one of four coarse crime-related categories: **robbery**, **theft**, **assault**, or **normal**. It uses OpenCV to detect pedestrian counts and motion intensity, aggregates statistics, runs a rule-based classifier, and generates a short natural-language summary.

> ⚠️ **Important:** This project is intended for experimentation and prototyping only. The heuristics are not a replacement for a properly trained action-recognition system and should not be used in production-critical decision making.

## Getting started

1. Create and activate a Python 3.10+ virtual environment. Example using `venv`:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place any `.mp4`, `.mov`, `.mkv`, or `.avi` file in the project root or inside the `videos/` directory.

## Running the analyzer

```bash
python main.py                  # automatically picks the first video it finds
python main.py -v my_clip.mp4   # analyze a specific file
python main.py -v videos/clip.mp4 --sample-rate 4 --dump-stats reports/clip.json
```

Arguments:
- `--sample-rate`: sampled frames per second (default `3`). Higher values improve accuracy at the cost of longer runtimes.
- `--max-frames`: optional hard limit on processed frames.
- `--dump-stats`: path to a JSON file containing raw per-frame stats, aggregated features, and class probabilities.

## How it works

1. **Frame sampling:** `src/video_processing.py` reads the video with OpenCV, samples frames at the requested rate, counts visible people through the default HOG-based pedestrian detector, and estimates motion using optical flow.
2. **Feature aggregation:** `src/features.py` summarizes the sampled frames (average/peak motion, crowd ratios, calm ratios, etc.).
3. **Heuristic classification:** `src/classifier.py` combines those features to score four activity classes.
4. **Summary generation:** `src/summarizer.py` reports the predicted class, statistical context, and a human-readable note.

## Project layout

```
BTP/
├── main.py                # Entry point that wires the entire pipeline together
├── requirements.txt
├── README.md
├── videos/                # Drop your videos here (kept empty via .gitkeep)
└── src/
    ├── __init__.py
    ├── classifier.py      # Heuristic scoring logic
    ├── features.py        # Aggregation of per-frame stats
    ├── file_utils.py      # Helpers to locate input videos
    ├── summarizer.py      # Turns predictions into readable text
    └── video_processing.py# Frame sampling, motion & person detection
```

## Next steps

- Replace the heuristic classifier with a learned model (e.g., 3D CNN, transformer, or CLIP-based action recognizer).
- Expand the training labels beyond four classes and supply real annotated footage.
- Hook the pipeline into a message queue or REST API for batch processing.
# BTP
