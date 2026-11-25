
---
title: Qwen DPO Alignment Experiment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Qwen2.5-0.5B: Before vs After DPO

This Space compares the original Qwen2.5-0.5B base model with two DPO fine-tuned versions.

## Models

- Base: `Qwen/Qwen2.5-0.5B`
- DPO beta 0.1: `rashidiqra/qwen2.5-0.5b-dpo-beta01`
- DPO beta 0.5: `rashidiqra/qwen2.5-0.5b-dpo-beta05`

## Features

- Generate a response from the base model.
- Generate a response from either DPO model.
- Compare all three model responses side by side.
- Adjust maximum new tokens and temperature.

## Dataset

`trl-lib/ultrafeedback_binarized`

The DPO model repositories must be public or accessible to this Space.
This application requires CUDA-capable Hugging Face Space hardware.
