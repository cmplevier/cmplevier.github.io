---
layout: post
title: "Testing Shortcut Learning with Marked MNIST1D"
date: 2026-05-19
---

<style>
  .post-content {
    --ink: #172026;
    --muted: #5e6b73;
    --line: #d9e3e2;
    --paper: #f7faf9;
    --teal: #128277;
    --coral: #c94b4b;
    --amber: #b7791f;
    color: var(--ink);
  }

  .post-content .lead {
    font-size: 1.18rem;
    line-height: 1.65;
    color: #263238;
    margin-top: 1.1rem;
  }

  .post-content h2 {
    margin-top: 2.5rem;
  }

  .post-content h3 {
    margin-top: 2rem;
  }

  .post-content figure {
    margin: 1.65rem 0 2rem;
    padding: 0;
  }

  .post-content figure img {
    display: block;
    width: 100%;
    height: auto;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--paper);
  }

  .post-content figcaption {
    margin-top: 0.65rem;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.45;
  }

  .post-content .takeaway {
    border-left: 4px solid var(--teal);
    background: #f1f8f7;
    padding: 1rem 1.15rem;
    margin: 2rem 0;
    color: #213332;
  }

  .post-content .rq {
    border-left: 4px solid var(--coral);
    background: #fdf3f3;
    padding: 1rem 1.15rem;
    margin: 1.7rem 0;
    color: #3a2626;
  }

  .post-content table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.92rem;
    margin: 1.4rem 0 0.4rem;
  }

  .post-content th,
  .post-content td {
    border: 1px solid var(--line);
    padding: 0.45rem 0.6rem;
    text-align: center;
  }

  .post-content th {
    background: var(--paper);
  }

  .post-content td:first-child,
  .post-content th:first-child {
    text-align: left;
  }

  .post-content .table-caption {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0.4rem 0 2rem;
  }
</style>

<p class="lead">
Deep classifiers often reach high accuracy by exploiting an unintended cue that happens to correlate with the label, instead of the feature I actually care about. This failure mode is called <strong>shortcut learning</strong>, and it is dangerous precisely because it is invisible on a standard test set: the model looks accurate until the cue disappears or changes.
</p>

This blog introduces a **control dataset** for detecting that failure mode. I take MNIST1D and inject a **marker**: a small, fixed signal whose position is determined by the label. Because I build the shortcut myself, I know the ground truth, so any model that relies on the marker can be caught simply by removing or reversing it.

<div class="rq">
  <strong>Research question.</strong> When trained on marked data, does a classifier learn the digit shape, or does it take the shortcut and learn the marker?
</div>

**Why a control dataset (and not just another in-the-wild study)?** Prior work (see the background section) already shows shortcut learning is real and damaging in deployed systems. Those studies *discover* the problem; they do not give a cheap, fully controlled way to test whether a specific model or evaluation pipeline is fooled by a shortcut. That is the gap this dataset fills: a lightweight, ground-truth-known probe where the shortcut is planted on purpose, so a single test (reverse the marker) gives a clean yes/no answer to the research question above.

**Code and data.** Dataset generation code: [github.com/cmplevier/toy_dataset_FRMDL](https://github.com/cmplevier/toy_dataset_FRMDL). Blog post code: [github.com/cmplevier/cmplevier.github.io](https://github.com/cmplevier/cmplevier.github.io). Live blog post: [cmplevier.github.io/2026/05/19/shortcut-learning-mnist1d.html](https://cmplevier.github.io/2026/05/19/shortcut-learning-mnist1d.html).

## Background: shortcut learning

A useful overview is given by [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780), which explains how networks can look successful yet fail under a small distribution shift because they rely on unintended correlations rather than meaningful features.

A classic example is the cow-on-grass case from [Recognition in Terra Incognita](https://arxiv.org/abs/1807.04975): a network learns that "grass" predicts "cow" and then fails on cows photographed elsewhere. The same pattern appears in medical imaging, where a [pneumonia classifier](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683) and a [COVID-19 classifier](https://doi.org/10.1038/s42256-021-00338-7) learned hospital-specific image artifacts instead of medically meaningful features, and collapsed on scans from unseen hospitals. [Lapuschkin et al.](https://doi.org/10.1038/s41467-019-08987-4) call this "Clever Hans" behaviour: a model that appears competent but secretly relies on an easily identifiable cue that correlates with the label.

In every case the shortcut is only discovered *after* the model fails. A control dataset flips this around: by planting a known marker, I can test for shortcut reliance before trusting any accuracy number.

## Standard MNIST1D

MNIST1D is a one-dimensional adaptation of MNIST. Instead of a 28 x 28 grayscale image, each sample is a 40-dimensional signal, synthetically generated from digit-shaped templates and modified with random translation, scaling, noise, and background patterns. These transformations keep the task nontrivial while remaining lightweight. Like MNIST, it has ten classes (digits 0-9). The figure below shows samples; with no marker present, a classifier must use the signal shape.

<figure>
  <img src="{{ '/assets/images/regular_data.png' | relative_url }}" alt="Examples from the standard MNIST1D dataset shown as one-dimensional digit signals">
  <figcaption>Standard MNIST1D samples. Each digit is a 40-dimensional 1-D signal. No marker is present, so the only usable information is the shape of the signal.</figcaption>
</figure>

## Constructing the marked dataset

I append 10 extra dimensions to each sample and place the marker in the final dimension, keeping it cleanly separated from the original 40-dim signal. The marker position encodes the label group: labels 0-4 get a marker in one location, labels 5-9 in another, and the position is fixed within each group. Because the marker sits at a fixed, label-correlated position, it is far easier to read than the digit shape, so a classifier is strongly incentivised to use it. (Note: the marker is enlarged in the figure below for visibility only.)

<figure>
  <img src="{{ '/assets/images/marked_data.png' | relative_url }}" alt="Marked MNIST1D examples where labels 0 through 4 use one marker position and labels 5 through 9 use another marker position">
  <figcaption>Marked MNIST1D samples. The original 40 signal dimensions are unchanged; the appended marker (enlarged here for visibility only) encodes the label group. A classifier can now reach high accuracy by reading the marker alone, ignoring the digit shape.</figcaption>
</figure>

## The reversed-marker test

To turn the marker into a yes/no test of shortcut reliance, I create a third dataset in which the marker positions of the two label groups are swapped. The digit signals are identical; only the marker-label relationship is inverted.

<figure>
  <img src="{{ '/assets/images/marked_reversed_data.png' | relative_url }}" alt="Reversed marker MNIST1D examples where the marker positions are swapped between label groups">
  <figcaption>Reversed-marked samples. The digit signals are unchanged, but the marker positions of the two label groups are swapped. A model that learned the shape is unaffected; a model that learned the marker will mislabel almost every sample.</figcaption>
</figure>

<div class="takeaway">
  <strong>Core idea:</strong> a shape-learner should be unaffected by the reversal, whereas a marker-learner should perform well on marked test data but collapse when the marker is reversed.
</div>

## Classification experiments

I train four classifiers (logistic regression, MLP, CNN, GRU) and compare four conditions, each chosen to isolate one question:

- **Standard → standard:** baseline accuracy with no marker anywhere.
- **Marked → standard:** does the marker hurt once it is removed at test time?
- **Marked → marked:** can the model exploit the marker when it is present?
- **Marked → reversed:** does the model *rely* on the marker?

| Case | Logistic | MLP | CNN | GRU |
|---|---|---|---|---|
| Standard train → standard test | 31.5 | 65.8 | 94.4 | 97.6 |
| Marked train → standard test | 27.5 | 40.5 | 54.5 | 81.1 |
| Marked train → marked test | 48.1 | 76.6 | 98.1 | 98.6 |
| Marked train → reversed-marked test | 0.0 | 5.9 | 2.8 | 8.1 |

<p class="table-caption">Test accuracy (%). Marker-rich models (CNN, GRU) are near perfect on marked test data, drop when the marker is removed, and collapse to near zero when it is reversed: the signature of a model classifying by marker rather than by digit shape.</p>

The table answers the research question. Models trained on marked data score highest on the marked test (CNN 98.1%, GRU 98.6%), drop when the marker is removed, and collapse to near zero on the reversed test (logistic 0.0%, CNN 2.8%, GRU 8.1%). A model that had learned the digit shape would be unaffected by reversing the marker; the collapse shows these models classified almost entirely by the marker.

### Shuffled-input sanity check

As a final sanity check, I repeated the standard MNIST1D experiment after shuffling the order of the input. Each standard sample is a 40-dimensional 1-D signal, and an input position is simply one slot in that vector (position 1 is the first signal value, position 40 the last). I apply one fixed random permutation to these 40 positions and use the same permutation for both the training and test sets. This reorders the entire signal, scrambling its local left-to-right structure while leaving the set of values unchanged. A fully connected model can still learn from the same values in their new fixed positions, but a CNN or GRU should suffer, because the nearby and sequential relationships they rely on have been destroyed.

| Case | Logistic | MLP | CNN | GRU |
|---|---|---|---|---|
| Standard train → standard test | 31.5 | 65.8 | 94.4 | 97.6 |
| Shuffled-standard train → shuffled-standard test | 31.5 | 66.2 | 60.5 | 52.8 |

<p class="table-caption">Sanity-check accuracy (%) after applying one fixed random permutation to the 40-dimensional input. Logistic regression and the MLP are nearly unchanged because they can learn from the same input values in their new fixed positions, while the CNN and GRU drop sharply because the permutation destroys the local and sequential structure of the 1-D signal.</p>

The results match this interpretation. Logistic regression and the MLP stay almost exactly where they were, because a fixed permutation only changes which weight connects to which input slot. The CNN and GRU, however, fall from 94.4% and 97.6% to 60.5% and 52.8%, showing that their high standard accuracy depends on the ordered structure of MNIST1D. This shuffled experiment is therefore not the main shortcut-learning test; it is an architecture sanity check, while the actual test for marker reliance remains the reversed-marker condition above.

## Conclusion

A planted, label-correlated marker is enough to make standard classifiers ignore the digit shape: they look excellent on a same-distribution test set yet score near zero the moment the shortcut is reversed. The take-home message is that a high test accuracy alone does not tell you *why* a model is right. When a spurious cue might exist in real data — a hospital tag, a watermark, a consistent background — a small control dataset like this one, where the shortcut is known by construction, gives a direct test for whether a model is taking it.