# Fashion Trend Forecasting with Machine Learning

A machine learning pipeline that analyzes Valentino runway collections across 8 seasons, identifies visual trends, and forecasts what the next season's collection will look like — including AI-generated images of the predicted looks.

---

## Overview

This project explores whether fashion trends can be extracted from runway imagery and used to predict future collections. Using Claude's vision API to analyze 617 runway photos across 8 Valentino seasons, it builds a structured dataset of colors, silhouettes, garments, fabrics, and aesthetics — then applies two independent forecasting methods to predict Valentino SS_2027.

The project was intentionally limited to a single designer (Valentino) to isolate trend evolution over time rather than conflating house-specific style signatures with seasonal shifts.

---

## Pipeline

```
Runway images (617 photos, 8 seasons)
        ↓
Claude Vision API — analyzes each image → structured JSON
        ↓
CSV dataset (colors, silhouette, garments, fabrics, patterns, aesthetic)
        ↓
Trend analysis — frequency distributions per season
        ↓
    ┌───────────────────────────┐
    ↓                           ↓
Statistical forecast        Claude API forecast
(weighted recency +         (reasoning over
linear trend projection)     trend summary)
    ↓                           ↓
    └───────────────────────────┘
                ↓
    Claude writes image generation prompts
                ↓
        DALL-E 3 generates images
                ↓
    SS_2027 predicted looks (two versions)
```

---

## Data

- **Designer:** Valentino
- **Seasons:** fw_2023, ss_2024, fw_2024, ss_2025, fw_2025, ss_2026, fw_2026, ss_2023
- **Total images:** 617 runway photos
- **Source:** Vogue Runway

Raw images are not included in this repository. To reproduce the dataset, download runway images for each season and place them in the corresponding `data/` subdirectory.

---

## Forecasting Methods

### Method 1: Claude API Forecast
The per-season trend summary is passed to Claude as a structured text prompt. Claude reasons about directional shifts — distinguishing SS-specific patterns from FW patterns — and produces a narrative forecast for SS_2027 along with a detailed image generation prompt.

### Method 2: Statistical Forecast
Two data-driven approaches run on SS seasons only (ss_2023 → ss_2024 → ss_2025 → ss_2026):

- **Weighted recency averaging** — computes frequency share of each value per season, weighted so recent seasons count more heavily toward the prediction
- **Linear trend projection** — fits a linear regression to each value's frequency share across the 4 SS seasons and projects forward to SS_2027, surfacing values that are growing even if not yet dominant

The two statistical methods are compared side by side in a bar chart saved to `outputs/`.

---

## Results

Both forecasting methods converge on a **black and gold palette** with **tailored, structured silhouettes** and a **haute couture / avant-garde aesthetic** as the dominant prediction for SS_2027. The linear trend projection additionally surfaces emerging signals like `modern luxury` and `androgynous` that weighted recency does not, reflecting values with upward momentum even if not yet at the top.

Generated forecast images are saved in `outputs/`:
- `forecast_claude_ss2027.png` — image from Claude API forecast
- `forecast_statistical_ss2027.png` — image from statistical forecast

---

## Setup

### Requirements
```
pip install anthropic openai pandas numpy matplotlib pillow requests
```

### API Keys
This project requires two API keys. Create the following files in the project root (both are gitignored):

- `fashionAPI.txt` — Anthropic API key
- `image-gen-API.txt` — OpenAI API key

### Folder Structure
```
Fashion-ML-Project/
├── FashionML_project.ipynb
├── fashionAPI.txt          ← gitignored
├── image-gen-API.txt       ← gitignored
├── data/
│   ├── fw_2023/
│   ├── ss_2023/
│   ├── fw_2024/
│   ├── ss_2024/
│   ├── fw_2025/
│   ├── ss_2025/
│   ├── fw_2026/
│   └── ss_2026/
└── outputs/
    ├── valentino_analysis_results.csv
    ├── statistical_forecast.png
    ├── forecast_claude_ss2027.png
    └── forecast_statistical_ss2027.png
```

### Running the notebook
Run cells in order. The notebook is organized into the following sections:
1. Setup & API key loading
2. Single image sanity check
3. Batch image analysis (all 617 images → CSV)
4. Trend analysis & visualization
5. Claude API forecast
6. Statistical forecast
7. Image generation

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| Anthropic Claude (Haiku) | Batch image analysis |
| Anthropic Claude (Sonnet) | Trend forecasting & prompt writing |
| OpenAI DALL-E 3 | Image generation |
| pandas | Dataset management |
| numpy | Linear trend projection |
| matplotlib | Trend visualization |
| Pillow | Image preprocessing |

---

## Limitations

- 8 seasons is a small sample for statistical forecasting — linear trend projections should be interpreted as directional signals rather than precise predictions
- Claude's image analysis uses free-text fields for silhouette and aesthetic, which introduces some inconsistency in vocabulary across images
- DALL-E 3 generates photorealistic fashion images but cannot fully replicate the specific construction details of haute couture
