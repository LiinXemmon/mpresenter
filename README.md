# mPresenter

**mPresenter** is an end-to-end **multilingual agentic framework** designed to transform static academic papers (PDFs) into presentation videos.


## ✨ Features

![mPresenter overview](assets/Figure_2.png)

* **🤖 Multi-Agent Collaboration**: Orchestrates 4 specialized agents (Planner, Reviewer, Coder, Interpreter) to plan, critique, and generate professional slides.
* **👁️ Cross-Lingual Interpretation**: A dedicated **Interpreter Agent** analyzes figures to explain visual elements that remain in the source language.
* **🎨 True-to-Layout Generation**: Writes executable **Beamer LaTeX** code and visually inspects rendered slides to ensure high readability.
* **📈 High Information Density**: Designed for effective knowledge transfer, achieving higher accuracy in QA benchmarks compared to prior systems.
* **⚡ Cost-Effective Efficiency**: Achieves the **lowest token consumption** among baselines and substantially reduces latency, making high-quality video generation affordable and scalable.

## 🎬 Demo Videos

**English:**

https://github.com/user-attachments/assets/bbc52e4c-b729-4dd7-8df1-738ba1abab21



**中文:**

https://github.com/user-attachments/assets/c7331a7e-c669-4b68-aaaa-106ecef4a28a




## 📊 mPreBench Dataset

![mPreBench topic distribution](assets/Figure_3.png)

An expert-curated multilingual benchmark designed to evaluate **Effective Information Transfer (EIT)** of Paper2Video systems.

* **📄 40 Academic Papers**:
    * 20 English Papers (10 from NeurIPS 2025, 10 from ACL 2025).
    * 20 Chinese Papers (from *Chinese Journal of Computers*).
    * Covers diverse topics: Vision, NLP, Graph, Security, RL, BioMed, and Systems.
* **❓ 1,600 Multilingual Questions**:
    * Each paper is paired with **8 expert-written multiple-choice questions**.
    * All questions are translated into **5 languages**: English, Chinese, German, Japanese, and Arabic.


### Evaluation Dimensions
The benchmark targets four core aspects of scientific communication:
* **Motivation**: Research context and gaps in related work.
* **Method**: Technical mechanisms and figure interpretation.
* **Experiment**: Experimental setup and result analysis.
* **Conclusion**: Key takeaways and supported claims.

## 🚀 Quick Start

System dependencies:
- TeX Live (XeLaTeX).
- Poppler (`pdftoppm`) or ImageMagick.
- Fonts: *Source Han Sans* and *Fira Sans*

1. Install dependencies with pixi:

```bash
pixi install
# Enable CosyVoice TTS
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
```

2. Set an LLM key:
   - OpenAI: set `llm.openai_api_key` in `config.json`.
   - Gemini: set `llm.gemini_api_key` in `config.json` or export `GEMINI_API_KEY` / `GOOGLE_API_KEY`.
3. Place your PDF at `input/paper.pdf` or pass `--source-pdf`.
4. Run:

```bash
pixi shell
python main.py --source-pdf input/paper.pdf --target-language English --output-root output
```

## 🐳 Docker Deployment


```bash
docker build -t mpresenter:latest .
```

Run:

```bash
mkdir -p input output cache
cp /path/to/paper.pdf input/paper.pdf
docker run --rm -it \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/config.json:/app/config.json" \
  -e GEMINI_API_KEY=YOUR_KEY \
  mpresenter:latest \
  --source-pdf /app/input/paper.pdf --target-language English --output-root /app/output
```


## ⌨️ CLI

Common command-line options:

- `--source-pdf`: override `source_pdf` (default `input/paper.pdf`)
- `--output-root`: override `output_root`
- `--cache-root`: override `cache_root`
- `--target-language`: override `target_language` (default: English)
- `--planner-note`: override `planner_note`
- `--backbone`: override all LLM model names with a single model

Other configuration is defined in `config.json`.

Session cache IDs are derived from the input PDF filename (stem), so re-running with the same PDF name reuses the same cache directory under `cache/`.

## 📂 Outputs

Cache directory (`cache/<session_id>/`) key files and folders:
- `steps_status.json`: pipeline step status for resume.
- `run.log`: main run log.
- `final_outline.json`: finalized outline JSON.
- `slides_manifest.json`: slide-level manifest (image path + note).
- `slides.tex` and `slides.pdf`: merged Beamer source and PDF.
- `slides_llm/`: slide PNGs for LLM review.
- `slides/`: slide PNGs for video synthesis.
- `final_scripts.json`: finalized narration scripts.
- `audio/`: synthesized audio clips per slide.

Final output:
- `output/<source_pdf_stem>.mp4`: final video.

## ⏱️ Efficiency Analysis

![Efficiency analysis](assets/Figure_5.png)

mPresenter maintains low latency and minimal API costs while producing superior video quality.

## 🙌 Acknowledgements

Thanks to these open-source projects:
- PaddleOCR (PP-DocLayout-L + OCR): https://github.com/PaddlePaddle/PaddleOCR
- Metropolis Beamer theme (mtheme): https://github.com/matze/mtheme
- FunAudioLLM/CosyVoice (TTS): https://github.com/FunAudioLLM/CosyVoice
