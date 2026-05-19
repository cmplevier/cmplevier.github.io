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

</style>

<p class="lead">
This control dataset tests whether a model is robust to spurious label-correlated artifacts. By adding artificial markers that correlate with the class label, the dataset creates a controlled "Clever Hans" setting: a model can achieve high accuracy by exploiting the marker, but such behavior would indicate shortcut learning rather than learning the intended visual concept.
</p>

## Standard MNIST1D

The standard MNIST1D dataset is a simplified one-dimensional adaptation of the original MNIST handwritten digit dataset. Instead of representing each digit as a 28 x 28 grayscale image, every sample in MNIST1D is encoded as a 40-dimensional sequence of values, forming a small one-dimensional signal. The dataset is synthetically generated from digit-shaped templates inspired by MNIST and then modified with several random transformations such as translation, scaling, added noise, and background patterns. These variations make the classification task nontrivial while remaining computationally lightweight. Like the original MNIST dataset, MNIST1D contains ten classes corresponding to the digits 0 through 9.

Some examples can be seen below.

<figure>
  <img src="{{ '/assets/images/regular_data.png' | relative_url }}" alt="Examples from the regular MNIST1D dataset shown as one-dimensional digit signals">
  <figcaption>Regular MNIST1D samples. Each digit is represented as a short one-dimensional signal instead of a two-dimensional image.</figcaption>
</figure>

## Adding Label-Correlated Markers

The adapted data used for testing this property was generated using modified code that adds a marker based on the label. Labels 0-4 have a marker at one location, while labels 5-9 have a marker at another location. Importantly, these markers remain in a constant position for each label group, which can be seen below. This was done by adding 10 additional dimensions to the dataset and placing the marker in the final dimension, making it clearly separated from the original signal.

Some examples can be seen below. Note that the actual marker is not as large as shown in the figures; the square was added purely for visualization purposes.

<figure>
  <img src="{{ '/assets/images/marked_data.png' | relative_url }}" alt="Marked MNIST1D examples where labels 0 through 4 use one marker position and labels 5 through 9 use another marker position">
  <figcaption>Marked MNIST1D samples. The original 40 signal dimensions remain intact, while the added marker dimensions create a shortcut that correlates with the label group.</figcaption>
</figure>

## Reversed Marker Test

For the final testing setup, we also created a dataset which reversed the marker locations, which can be seen below.

<figure>
  <img src="{{ '/assets/images/marked_reversed_data.png' | relative_url }}" alt="Reversed marker MNIST1D examples where the marker positions are swapped between label groups">
  <figcaption>Reversed-marker samples. The visual digit signal is unchanged in spirit, but the shortcut cue has been swapped between the two label groups.</figcaption>
</figure>

Because the markers are located at fixed positions based on the label, they strongly influence the final training process. The classifiers will likely overfit to the markers instead of learning the features corresponding to the actual digit shapes. This behavior aligns closely with the shortcut learning phenomena described in the literature discussed below.

<div class="takeaway">
  <strong>Core idea:</strong> if a classifier learns the marker instead of the digit shape, it should perform well when the marker-label relationship is present, but degrade when that relationship is removed or reversed.
</div>

## Shortcut Learning Background

A useful overview of shortcut learning is given by the paper <a href="https://arxiv.org/abs/2004.07780">Shortcut Learning in Deep Neural Networks</a> by Geirhos et al. (2020). The paper explains how deep neural networks can appear superficially successful while failing under slightly different circumstances because they rely on unintended correlations instead of meaningful features.

One example discussed is image classification of cows. A neural network may classify cows extremely well during training, yet fail when cows appear outside grassy environments. In this case, the network unintentionally learns that "grass" is a predictor for "cow," instead of learning the shape or appearance of the animal itself. This phenomenon was also explored in <a href="https://arxiv.org/abs/1807.04975">Recognition in Terra Incognita</a> by Beery et al.

Another example comes from medical imaging. Zech et al., in <a href="https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683">Variable Generalization Performance of a Deep Learning Model to Detect Pneumonia in Chest Radiographs</a>, demonstrated that a classifier trained on chest X-rays from a number of hospitals performed poorly on scans from unseen hospitals. The model had unintentionally learned to identify hospital-specific imaging characteristics instead of learning features associated with pneumonia itself.

Shortcut learning typically reveals itself through a discrepancy between the intended and actual learning strategy, often causing unexpected failure when the environment changes. Similar concepts can also be found in fields such as psychology and education.

Shortcuts can be understood as decision rules that exploit unintended features which are often not generalizable. Ideally, a classifier should learn the intended features of the task, such as the shape of an animal or the structure of a handwritten digit.

These shortcuts are often learned because exploiting them is significantly easier than learning the intended solution. This behavior can be traced back to the inductive bias of both the model and the dataset. Inductive bias refers to the assumptions that influence which solutions are learnable and how readily they can be learned. According to the literature, four important aspects determine inductive bias:

- Model architecture and structure
- Training data and experience
- Loss function
- Optimization procedure

The idea of "Clever Hans" behavior is also described in <a href="https://arxiv.org/abs/1902.10178">Clever Hans Predictors and the Importance of Separating Features from Artifacts</a>. In this setting, a model overfits by relying on easily identifiable properties of the data that happen to correlate with the correct labels, but which do not actually represent the intended features. In other words, the model exploits spurious correlations.

This is closely related to findings from <a href="https://scispace.com/pdf/ai-for-radiographic-covid-19-detection-selects-shortcuts-2oi74y6ji7.pdf">AI for Radiographic COVID-19 Detection Selects Shortcuts</a>. Similar to the pneumonia example, the models learned markers and dataset-specific artifacts instead of medically meaningful features, resulting in a clear performance drop when tested on data from unseen hospital systems. The paper highlights how this issue was present in many high-profile COVID-19 studies that relied on limited and non-representative datasets.

## Classification Experiments

To test this shortcut learning behavior, we then performed classification experiments using both marked and unmarked training and test datasets. The results clearly show that the classifiers fit onto the markers: models trained on marked data performed worse on regular test data than models trained on regular data. This demonstrates that the classifiers relied on the shortcut artifacts instead of learning the intended digit representations.

<figure>
  <img src="{{ '/assets/images/Classifier_results.png' | relative_url }}" alt="Classifier experiment results comparing marked and unmarked training and test datasets">
  <figcaption>Classifier results comparing marked and unmarked training and test datasets.</figcaption>
</figure>
