# AI Image → JPG <100KB Streamlit App

This version uses AI foreground segmentation instead of simply replacing
white pixels.

## Processing pipeline

1. Accept one/multiple images or a ZIP.
2. Correct EXIF orientation.
3. Convert source to RGB.
4. Run AI background segmentation.
5. Keep the detected foreground.
6. Replace the complete background with the configured color.
7. Encode as JPEG.
8. Iteratively reduce JPEG quality.
9. If necessary, reduce dimensions and continue.
10. Guarantee the output is strictly below 100 KB where possible.
11. Preserve filenames and ZIP folder structure.
12. Download individual JPGs or one ZIP.

## AI model

The app uses `bria-rmbg`, through `rembg`, as the default segmentation
model.

The model is downloaded/cached on first use.

IMPORTANT LICENSE NOTE:
The BRIA RMBG-2.0 weights are available under a non-commercial
CC BY-NC 4.0 license according to the model card. For commercial
deployment, use an appropriately licensed model/API or obtain the
required commercial license.

## Configuration

At the top of `app.py`:

```python
BACKGROUND_COLOR = "#7CA1BB"
MAX_FILE_SIZE_KB = 100

AI_MODEL = "bria-rmbg"

INITIAL_JPEG_QUALITY = 95
MIN_JPEG_QUALITY = 25
QUALITY_STEP = 5
RESIZE_STEP = 0.90
MIN_DIMENSION = 100
```

### Alternative models

`rembg` supports other segmentation models. For example:

```python
AI_MODEL = "u2net_human_seg"
```

can be useful when the images are specifically human photographs.

For general mixed photographs:

```python
AI_MODEL = "bria-rmbg"
```

is the default.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first AI inference can take longer because the model must be
downloaded and initialized. The model is cached and reused for
subsequent images.

## Commercial use

Check the exact license of the selected model before deploying this
inside a commercial organization. The BRIA RMBG-2.0 model card states
that the self-hosted weights are for non-commercial use unless a
commercial agreement is obtained.
