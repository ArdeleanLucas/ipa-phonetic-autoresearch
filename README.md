# ipa-phonetic-autoresearch

**Autonomous LLM-driven experimentation to hyper-specialize Wav2Vec2ForCTC models for precise IPA phoneme recognition of rare or dialect-specific patterns**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org)

**Research Question**
Can Karpathy's AutoResearch loop discover reproducible, linguistically meaningful improvements in a Wav2Vec2 CTC model's ability to transcribe a *single target phonetic pattern* into accurate IPA symbols, using only 100–2000 human-transcribed audio clips?

Drop your audio + exact IPA transcripts into the repo, launch the loop overnight, and wake up to a git log containing **only validated improvements** on targeted Phoneme Error Rate (PER).

## Features
- Native `Wav2Vec2ForCTC` + custom IPA tokenizer
- Targeted + weighted PER (articulatory-feature distance)
- Multiple loop variants (classic, config-driven, multi-agent)
- Full linguistic scaffolding (`DATA_STATEMENT.md`, `ETHICS.md`, qualitative notebook)
- Baselines, ablations, generalization tests, Praat export

## Quick Start
```bash
git clone https://github.com/ArdeleanLucas/ipa-phonetic-autoresearch.git
cd ipa-phonetic-autoresearch
uv sync
python prepare.py --help
python harness.py --model facebook/wav2vec2-base
```

See [`RESEARCH_IDEA.md`](RESEARCH_IDEA.md) for the full research design and detailed setup.

## Repository Structure
See [`RESEARCH_IDEA.md`](RESEARCH_IDEA.md).

## Limitations & Roadmap
- Best for modest datasets (100–2000 clips)
- Requires a capable LLM (Claude 3.5/4o or Grok recommended)
- Experimental — always validate final models with human phoneticians

## Citation
```bibtex
@misc{ipa-phonetic-autoresearch,
  author = {Lucas Ardelean},
  title  = {ipa-phonetic-autoresearch},
  year   = {2026},
  url    = {https://github.com/ArdeleanLucas/ipa-phonetic-autoresearch}
}
```
