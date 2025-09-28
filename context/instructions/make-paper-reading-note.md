you are tasked to read a paper and create a structured note summarizing its key aspects. Use the following template to ensure all important elements are captured. If user does not specify an output dir, save the note in dir `notes/<paper-name>/`, name the output markdown as `main-note.md`, and store any images in a subdir `figures/`. Ensure all image paths in the markdown are relative to the markdown file location.

formatting guide:
- for bulleted lists, format it like "- **Item**: description"

citation guide:
- to ground your note, include blockquote snippets from the paper with proper citations

---

below is the template

# [Paper Title]

## 0. Metadata
- **Full Title**: 
- **Authors**: 
- **Venue / Year**: 
- **Links**: PDF | Project Page | Code | Dataset | ArXiv/DOI
- **Keywords**: 
- **Paper ID (short handle)**: 

## 1. TL;DR (3–5 bullets)
- Problem in one line:
- Core idea in one line:
- Main contribution(s):
- Why it matters / where it applies:

## 2. Problem & Motivation
- What problem is being solved? Why now? Prior limitations/gaps:
- Assumptions and scope (what is in/out):

## 3. Key Ideas & Contributions (Condensed)
- 
- 
- 

## 4. Method Overview
- One-paragraph summary of the approach and how components interact.
- High-level data flow or pipeline description.

## 5. Interface / Contract (Inputs & Outputs)
- Inputs (types, shapes, modalities, constraints):
- Outputs (types, interpretations, constraints):
- Control/conditioning signals (if any):
- Required pre/post-processing steps:

## 6. Architecture / Components
- Components and their responsibilities:
  - [Component A]: brief description
  - [Component B]: brief description
- How components connect (diagram reference):

## 7. Algorithm / Pseudocode (Optional)
```text
# Pseudocode outline for core loop or training/inference
```

## 8. Training Setup
- Data: sources, size, splits, preprocessing:
- Objective(s) / loss functions:
- Model sizes / parameters:
- Hyperparameters & schedules:
- Compute budget (GPUs, hours):
- Training tricks & stabilization notes:

## 9. Inference / Runtime Behavior
- Inputs required at inference and their formats:
- Control knobs (temperature, steps, search, actions):
- Latency / throughput notes:
- Failure modes observed at inference:

## 10. Experiments & Results
- Benchmarks, datasets, and tasks evaluated:
- Metrics:
- Baselines and comparisons:
- Headline results (with pointers to tables/figures):

## 11. Ablations & Analysis
- What design choices matter most:
- Scaling trends:
- Sensitivity analyses:

## 12. Limitations, Risks, Ethics
- Stated limitations:
- Observed failure cases:
- Safety, bias, and ethical considerations:

## 13. Applicability & Integration Notes (Project-Focused)
- Where this could fit in our stack:
- Minimal viable integration / prototype plan:
- Dependencies or blockers:

## 14. Reproducibility Plan
- Checklist: data availability, code, configs, seeds:
- Reproduction steps at high level:
- Known gaps vs paper setup:

## 15. Related Work
- Closest prior work and how this differs:
- Historical context:

## 16. Open Questions & Follow-Ups
- 
- 

## 17. Glossary / Notation
- Symbols and their meanings:
- Important terms defined:

## 18. Figures & Diagrams (Optional)
- [Placeholder] Overview figure path:
- [Placeholder] Architecture diagram path:
- [Placeholder] Activity/flow diagram path:

## 19. BibTeX / Citation
```bibtex
@article{<key>,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```

---
