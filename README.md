# PumpkinV2: Optimized & Interactive Video Event Retrieval System

An end-to-end, highly scalable video event retrieval platform with improved temporal algorithms, a production-grade UI, and optimized back-end. Developed by the Saigon International University team for the AI Challenge HCMC 2024.

---

## Table of Contents

- [Features](#features)  
- [System Architecture](#system-architecture)  
- [Installation & Setup](#installation--setup)  
- [Quick Start](#quick-start)  
- [Configuration](#configuration)  
- [Folder Structure](#folder-structure)  
- [Core Components](#core-components)  
  - [1. Data Preprocessing](#1-data-preprocessing)  
  - [2. Query Processing & Temporal Retrieval](#2-query-processing--temporal-retrieval)  
  - [3. Reranking & Filters](#3-reranking--filters)  
  - [4. Web Server & UI](#4-web-server--ui)  
- [Usage Examples](#usage-examples)  
- [Performance & Benchmarking](#performance--benchmarking)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)  

---

## Features

- **High-precision video retrieval** via state-of-the-art CLIP-based embeddings  
- **Improved temporal search** using additive and multiplicative scoring  
- **Shot boundary detection** (PySceneDetect & Autoshot) with near-duplicate removal  
- **Fast, scalable storage**: Qdrant vector database + binary quantization  
- **Optimized asset delivery**: AV1 video encoding, AVIF thumbnails  
- **Interactive UI**: filter by metadata, transcripts, reranking, keyboard shortcuts  
- **RESTful API**: powered by a production-grade Gunicorn WSGI server  

---
---

<p align="center">
  <img src="system.gif" alt="System Architecture" width="700"/>
</p>

---
## System Architecture

```text
┌──────────────────┐      ┌─────────────┐      ┌─────────────┐
│  Video Dataset   │─┐    │  Data Prep  │─┐    │   Qdrant    │
│ (YouTube News)   │ ├─▶  │ • SBD       │ ├─▶  │ Vector +    │
└──────────────────┘ │    │ • VAD+STT   │ │    │ Metadata    │
                     │    │ • CLIP Emb. │ │    └─────────────┘
                     │    └─────────────┘ │
                     │                    │    ┌────────────────┐
                     │    ┌─────────────┐ │    │  Web Server    │
                     └───▶│  Query/API  │ └─▶  │ • Flask +      │
                          │ • Temporal  │      │   Gunicorn     │
                          │ • Rerank    │      └────────────────┘
                          └─────────────┘            │
                                                     ▼
                                            ┌────────────────┐
                                            │  Frontend UI   │
                                            │ • Flask Jinja  │
                                            │ • JS/CSS       │
                                            └────────────────┘
```

---

## Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/phamgiakiet273/Pumpkin_AIC_2024.git
   cd Pumpkin_AIC_2024
   ```

2. **Create a Python 3.9+ virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare external services**  
   - **Qdrant**  
     Install and run locally or point to a managed Qdrant instance.  
   - **(Optional) Distributed file store**  
     Configure NFS or another solution to host static assets if needed.

5. **Download / extract** your video dataset into `static/videos/` and thumbnails into `static/thumbnails/`.

---

## Quick Start

Once your environment and data are ready:
```bash
# For a local, private instance:
python server.py

# For a public-facing deployment:
python server_public.py
```
By default, the UI and API will be served on `http://localhost:8000`.

---

## Configuration

All tunable parameters live in `utils/config.py` (or environment variables):

- **`QDRANT_URL`** – Vector database endpoint  
- **`EMBEDDING_MODELS`** – Which CLIP variants to load (`SIGLIP`, `DFN5B`, etc.)  
- **`SBD_METHOD`** – Shot boundary detector (`pyscenedetect` or `autoshot`)  
- **`TEMPORAL_WINDOW`** – Δ-window in seconds for temporal aggregation (default 1500s)  
- **`GUNICORN_WORKERS`** – Number of WSGI worker processes  

---

## Folder Structure

```
├── api/                  # RESTful endpoints
├── base_2023/            # PumpkinV1 compatibility & utilities
├── libs/                 # Third-party wrappers
├── models/               # Embedding & ASR model load/save code
├── server.py             # Main application (private)
├── server_public.py      # Public-facing deployment
├── static/               # Videos, thumbnails, CSS/JS assets
├── templates/            # Jinja2 page templates
├── ui/                   # React/Vue/TS front-end (if applicable)
├── utils/                # Utility scripts (SBD, VAD, config)
├── requirements.txt      # Python dependencies
└── README.md             # ← You are here
```

---

## Core Components

### 1. Data Preprocessing

- **Shot Boundary Detection (SBD)**  
  - *PySceneDetect* for general content  
  - *Autoshot* fine-tuned on AIC 2022 for news programs  
- **Duplicate removal** via perceptual hash matching  
- **Voice Activity Detection & STT**  
  - *Pyannote* for segmenting speech  
  - *Wav2Vec2* fine-tuned on VLSP for Vietnamese transcripts  
- **Visual-Text Embeddings**  
  - *SIGLIP* for broad visual concepts  
  - *DFN5B* for robustness on longer, complex queries  
- **Vector Storage** in Qdrant with binary quantization for 32× compression and 27× search speedup

### 2. Query Processing & Temporal Retrieval

- **Single-scene search** using cosine similarity on CLIP embeddings  
- **Temporal multi-scene search**  
  - **Additive**: aggregates cosine scores across time windows  
  - **Multiplicative**: amplifies sequences of dense matches  
- **Temporal formula**  
  > Y(v,F,Q) = Σᵢ X(v, fᵢ ∈ [f + iΔ, f + (i+1)Δ], qᵢ)  
  > or  
  > Πᵢ X(v, fᵢ ∈ …)  

### 3. Reranking & Filters

- **Shot Collection Rerank** groups shots by story segment  
- **Color Sort Rerank** (step color sorting) for visual coherence  
- **Perceptual Hash Rerank** to bring near-duplicates together  
- **Metadata & Transcript Filters** for fine-grained narrowing

### 4. Web Server & UI

- Backend: **Flask** + **Gunicorn** (production WSGI)  
- Frontend: Jinja2 templates + minimal JS for interactivity  
- **Features**  
  - Model selection (SIGLIP / DFN5B)  
  - Voice-to-text query  
  - Adjustable thumbnail size & result count  
  - Keyboard shortcuts & pop-up preview  
  - Transcript toggling for speed  

---

## Usage Examples

1. **Textual KIS** – find “heavy snowfall obscuring cars”  
2. **Visual KIS** – upload a frame to locate its sequence  
3. **VQA** – count soccer players in a penalty scene using temporal search  

> Average retrieval + render time: **≈800 ms** for top-200 results, even on modest hardware.

---

## Performance & Benchmarking

On the AI Challenge HCMC 2024 dataset (292 h of video, 762 654 keyframes):

- **94 % accuracy** on qualifying rounds  
- **Top-10** finalist ranking  
- **< 1 min** average answer time in finals  
- **Vector search**: ~50 ms per query  

---

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/my‐enhancement`)  
3. Commit your changes (`git commit -am 'Add some feature'`)  
4. Push to the branch (`git push origin feature/my‐enhancement`)  
5. Open a Pull Request  

---

## References

- “An Optimized And Interactive Video Event Retrieval System With An Improved Temporal Algorithm,” K. Phạm Gia Kiet et al., AI Challenge HCMC 2024 – details of PumpkinV2’s design and benchmarks.  
- Qdrant: An open-source vector database – used for efficient high-dimensional search.  
- PySceneDetect, Autoshot, Pyannote, Wav2Vec2, SIGLIP, DFN5B – key libraries/models leveraged.  

---

*Built with ❤️ at Saigon International University.*
