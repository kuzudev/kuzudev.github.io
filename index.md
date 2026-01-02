---
title: ML Notes
layout: default
---

<div class="hero">
  <h1 class="hero__title">ML Notes</h1>
  <p class="hero__lead">
    Short research notes on computer vision, multimodal models, and practical ML engineering.
  </p>

  <div class="hero__links">
    <a class="btn" href="{{ '/articles/' | relative_url }}">Browse notes</a>
    <a class="btn btn--ghost" href="{{ '/articles/about.md' | relative_url }}">About</a>
    <a class="btn btn--ghost" href="https://github.com/<USERNAME>">GitHub</a>
  </div>
</div>

## Tracks

<div class="grid">
  <a class="card" href="{{ '/articles/vit.md' | relative_url }}">
    <div class="card__title">Vision Transformer → Open-Vocab Segmentation</div>
    <div class="card__desc">ViT basics, attention maps, CLIP and CLIP-based segmentation.</div>
  </a>

  <a class="card" href="{{ '/articles/siglip.md' | relative_url }}">
    <div class="card__title">Image Embeddings</div>
    <div class="card__desc">SigLIP, ViT-SO400M, AIMv2, Florence-2, DINOv2/v3.</div>
  </a>

  <a class="card" href="{{ '/articles/embs_choose.md' | relative_url }}">
    <div class="card__title">Choosing an embedding model</div>
    <div class="card__desc">Practical criteria: retrieval, speed, robustness, finetuning.</div>
  </a>
</div>

## Quick index

<div class="two-col">

<div>
  <h3>Transformers & open-vocab segmentation</h3>

  <ul>
    <li><a href="{{ '/articles/vit.md' | relative_url }}">Vision Transformer — Basics</a></li>
    <li><a href="{{ '/articles/attention_maps.md' | relative_url }}">Vision Transformer — Attention maps</a></li>
    <li><a href="{{ '/articles/clip.md' | relative_url }}">CLIP — Basics</a></li>
    <li><a href="{{ '/articles/sclip.md' | relative_url }}">SCLIP — upgrading CLIP for segmentation</a></li>
  </ul>
</div>

<div>
  <h3>Image embeddings</h3>

  <ul>
    <li><a href="{{ '/articles/siglip.md' | relative_url }}">SigLIP</a></li>
    <li><a href="{{ '/articles/vit_so400m.md' | relative_url }}">ViT-SO400M</a></li>
    <li><a href="{{ '/articles/aimv2.md' | relative_url }}">AIMv2</a></li>
    <li><a href="{{ '/articles/florence2.md' | relative_url }}">Florence-2</a></li>
    <li><a href="{{ '/articles/dinov2.md' | relative_url }}">DINOv2</a></li>
    <li><a href="{{ '/articles/dinov3.md' | relative_url }}">DINOv3</a></li>
  </ul>
</div>

</div>

<hr/>

<p class="muted">
If you want RSS and “latest posts”, we can convert <code>articles/*.md</code> into real posts under <code>_posts/</code> later — but this layout already looks academic and clean.
</p>
