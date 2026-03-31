# neurodyad_task

GSoC 2026 pre-task submission for the NeuroDyads project (ML4SCI).  
Brain-to-brain decoder using CEBRA on simultaneous EEG from a conversational dyad.

## What this is

This repo contains my solution to the NeuroDyads GSoC 2026 pre-task. The task involved taking two raw EEG recordings from a single dyad (one speaker, one listener) having a conversation, preprocessing them, learning a joint low-dimensional embedding using CEBRA, and interpreting what the model found.

The broader NeuroDyads project studies how two brains coordinate during natural conversation using simultaneous EEG (hyperscanning). 

---

## The task

Four parts:

**Part 1 — Preprocessing**
- Segment data by DIN1 markers: positive affect (DIN1_1 → DIN1_2), negative affect (DIN1_3 → EOF)
- Remove VREF channel (channel 65)
- Bandpass filter 1–40 Hz
- ICA artifact removal with documented component rejection reasons
- PSD before/after ICA comparison for one participant

**Part 2 — CEBRA Embedding**
- Concatenate two preprocessed 64-channel datasets into a T×128 joint matrix
- Z-normalise each channel independently
- Train CEBRA with 3D output, labels 0 (positive) / 1 (negative)
- Report KNN decoding accuracy and goodness-of-fit
- Run a shuffled-data control (shuffle Participant B's time axis only)

**Part 3 — Interpreting the Embedding**
- Describe the geometry: clusters, transitions, outliers, what they suggest
- Explain what the shuffled control result tells you about what CEBRA learned

**Part 4 — Critical Reflection**
- Identify the single biggest specific limitation of this analysis
- Describe what you'd do differently with more time and more data


---

## What I did and key decisions

**Segmentation**
Both files had 3 DIN1 markers but with ~0.2s clock drift between them. I cropped each file using its own marker times rather than a shared average, to avoid introducing misalignment at the sample level. DIN1_2 and DIN1_3 were only ~0.27s apart — this is just the block boundary marker, not a rest period, so I cropped right at the markers.

**Montage**
The EDF files had no electrode coordinates embedded (common with EGI exports). Channel names were `EEG 1, EEG 2...` but the GSN-HydroCel-64_1.0 montage expects `E1, E2...` — renamed before assigning.

**Filtering**
1–40 Hz bandpass before ICA. The 1 Hz highpass is the important one: slow drifts below 1 Hz can look like independent components to ICA and waste decomposition capacity. The hardware filter was already at 0.1–45 Hz so this just tightens it slightly.

**ICA**
20 components, FastICA. Fit separately per participant. Components rejected based on both topography and time course inspection:

| Participant | Rejected | Reason |
|-------------|----------|--------|
| A | ICA002, ICA003 | Eye blinks — uniform frontal field, large spike time course at same moments |
| A | ICA005 | Electrode drift — focal right temporal, slow wandering signal |
| A | ICA007 | Saccade — left-right asymmetric frontal, burst time course |
| A | ICA014 | Electrode noise — edge-focal topography, implausibly slow oscillation |
| B | ICA000 | Electrode noise — flat then irregular, focal left edge |
| B | ICA005 | Electrode pop — sudden large burst onset mid-recording |
| B | ICA007 | Muscle (EMG) — sustained high-frequency noise, focal frontal-left |
| B | ICA008 | Electrode noise — sudden burst onset |
| B | ICA009 | Electrode noise — near-flat with spikes, focal right edge |

**CEBRA**
`offset10-model` (40ms receptive field at 250Hz), 3D output, `time_delta` conditional, 5000 iterations. Shuffled control breaks inter-brain alignment by permuting only B's columns — preserves each brain's own dynamics, destroys their correspondence.

---

## Results

| | KNN Accuracy | Final InfoNCE Loss |
|---|---|---|
| Main model | 0.975 ± 0.001 | 5.716 |
| Shuffled control | 0.552 ± 0.007 | 6.235 |
| Chance | 0.510 | — |

The embedding formed a horseshoe/crescent shape with two compact clusters (positive and negative affect) connected by a smooth curved arc. The near-total collapse of accuracy in the shuffled control (0.975 → 0.552) confirms the structure was driven by inter-brain temporal coupling, not just each brain independently tracking the conversation content.

---

## Main limitation

With N=1 dyad, genuine inter-brain coupling can't be separated from both participants independently responding to the same shared conversation (speech, prosody, emotional content). The shuffled control rules out temporal coincidence but doesn't rule out shared stimulus tracking — A's individual response to the conversation is still intact in the unshuffled half, and the residual above chance (0.552 vs 0.510) is consistent with exactly this.

The fix is a pseudo-dyad control: pair A from this dyad with B from a completely different dyad who had a different conversation. Impossible with N=1.

---

## Setup

```bash
pip install mne cebra torch scikit-learn matplotlib numpy
```

Open `neurodyads_pretask.ipynb` in Jupyter or Colab. Update the two file paths at the top to point to your EDF files.

---
## Data

Raw EEG recordings are not included in this repository. Data was provided as part of the GSoC 2026 ML4SCI NeuroDyads pre-task and is not publicly available.

---

## References

- Schneider et al. (2023). Learnable latent embeddings for joint behavioural and neural analysis. *Nature*.
- Roca et al. (2023). Cross-entropy as a framework for comparing embedding distributions. *Cell Reports Methods*.
- ML4SCI NeuroDyads 2025 pilot: [github.com/ML4SCI/NeuroDyads](https://github.com/ML4SCI/NeuroDyads)
