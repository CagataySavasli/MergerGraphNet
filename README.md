# MergerGraphNet

**MergerGraphNet** is a research project for forecasting **Mergers and Acquisitions (M&A)** from financial reports.  
The project combines **benchmark models** and **novel approaches** to improve prediction performance.  

This repository contains the full code and experiments from the paper:  
*Predicting Mergers and Acquirers from Text via Deep Learning*.

---

## Overview

Mergers and Acquisitions are rare but important events in financial markets.  
We use both **text data** (10-K and 10-Q MD&A sections) and **quantitative fundamentals** to predict if a company will merge in the next quarter.  

Our methods include:
- **Statistical baselines** (Odds Ratio, Word Frequency)
- **Embedding-based models** (Word2Vec, BERT)
- **Hierarchical BERT models**: RoBERT, ToBERT
- **Graph-based model**: GoBERT (our novel approach)
- **Hybrid models**: combining text embeddings with financial fundamentals
- **Generative AI**: Gemini 2.5 (zero-shot forecasting with explanations)

---

## Key Contributions
- Provide a **benchmark** of traditional, embedding, and generative approaches for M&A forecasting.  
- Introduce **GoBERT**, a **graph over BERT** model for long financial texts.  
- Show the value of **hybrid models** that combine textual and quantitative data.  
- Explore **LLM reasoning** with Gemini to give explainable predictions.  

---

## Dataset

- **Text Data**: MD&A sections from **10-K and 10-Q filings** (2010–2021).  
- **Merger Labels**: From Thomson Reuters M&A dataset.  
- **Financial Fundamentals**: From WRDS (IBES).  

Total: **8,905 reports**, with ~17% merger cases.  

---

## Installation

```bash
git clone https://github.com/CagataySavasli/MergerGraphNet.git
cd MergerGraphNet
poetry install
```

---

## Contributors

- [Ahmet Cagatay Savasli](https://cagataysavasli.github.io/) – Methodology, Software, Writing
- [Emre Sefer](https://seferlab.github.io/) – Conceptualization, Methodology, Writing
