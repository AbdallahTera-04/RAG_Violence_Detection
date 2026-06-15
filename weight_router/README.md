# GuardianEye Model Weight Router

This package selects one of the three `E_full_qgf_fixed` checkpoints by estimating
how closely an input video resembles each checkpoint's training domain.

It keeps two concepts separate:

- **Domain similarity** chooses `HF`, `NTU`, or `RLVS`.
- **Violence confidence** is produced later by the selected classifier and is
  compared with that dataset's decision threshold.

## Checkpoints and thresholds

| Dataset | Selected checkpoint | Raw-output threshold | Test macro F1 |
|---|---|---:|---:|
| HF | `train_output_HF/runs_hf/E_full_qgf_fixed/best.pt` | 0.70 | 0.9400 |
| NTU | `train_output_NTU/runs_ntu/E_full_qgf_fixed/best.pt` | 0.08 | 0.8720 |
| RLVS | `train_output_RLVS/runs_rlvs/E_full_qgf_fixed/best.pt` | 0.21 | 0.9750 |

The router validates these thresholds against each checkpoint folder's
`test_metrics.json` at startup.

## Routing signals

The router samples the input video and measures duration, FPS, aspect ratio,
brightness, motion intensity, camera motion, and scene-change rate. It combines:

- video-feature similarity: 60%
- VLM/user context similarity: 25%, when supplied
- explicit specification similarity: 15%, when supplied

Missing context or specifications are excluded and the available weights are
renormalized. Dataset assumptions and their provenance live in
`model_weight_router/profiles.json`. Violence-action words such as `fight` are
not routing signals because they describe all three datasets. Dataset-specific
domain terms count more than terms shared by multiple profiles.

The returned score is intentionally named `heuristic_domain_similarity`. It is
an explainable routing estimate, not a statistically calibrated probability.

## Usage

From the `Model_Types` directory:

```powershell
python -m model_weight_router VIDEO.mp4 `
  --context "Fixed CCTV camera overlooking a public street" `
  --specifications '{"source":"cctv","camera":"fixed"}'
```

Programmatic routing:

```python
from model_weight_router import ModelWeightRouter

router = ModelWeightRouter(r"D:\RAG_Graduation_Translator\Model_Types")
decision = router.route_video(
    "uploaded_video.mp4",
    context=vlm_overview,
    specifications={"source": "cctv", "camera": "fixed"},
)

classifier = load_classifier(decision.selected_checkpoint)
violence_probability = classifier(...)
verdict = violence_probability >= decision.decision_threshold
```

Only the selected checkpoint should be loaded into GPU memory. The similarity
router itself does not load any checkpoint.

## Better similarity after obtaining training videos

The current profiles distinguish the public datasets from the exact
distributions used to train these checkpoints:

- **HF:** balanced, homogeneous hockey-broadcast clips.
- **NTU:** the project's balanced five-second window conversion of NTU
  CCTV-Fights, including hard negatives from the same source videos.
- **RLVS:** the project's balanced short-clip split, with documented scene
  re-cuts and a motion-presence shortcut.

The training videos are not present in `Model_Types`, so visual feature centers
remain documented estimates. Once the exact training-video folders are
available, build measured profiles with:

```powershell
python -m model_weight_router.profile_builder D:\path\to\HF\train `
  --output hf_empirical_profile.json
```

Run this for HF, NTU's generated five-second training windows, and RLVS. Replace
the corresponding `feature_profile` blocks with the generated values, then tune
the 60/25/15 routing weights on a separate labeled routing-validation set.

## Dataset sources

- Hockey Fight: Nievas et al., *Violence Detection in Video Using Computer
  Vision Techniques*, DOI `10.1007/978-3-642-23678-5_38`.
- NTU CCTV-Fights: [comparison and dataset usage paper](https://arxiv.org/abs/2205.11394)
  and the project's local conversion documentation.
- RLVS: [dataset page](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
  and the project's local integrity audit.

## Tests

```powershell
python -m unittest discover -s model_weight_router/tests -v
```
