# Read Me First For Qwen

## Can Qwen Perform The Routing Task?

`Qwen2.5-VL-3B-Instruct-AWQ` is applicable and useful for the model-routing
task. It can inspect an uploaded video and describe its visual domain, recording
style, environment, and clip structure.

Qwen should complement the existing weight router. It should not directly
replace the router or independently decide which checkpoint must be used.

## Recommended Architecture

```text
Uploaded video
  -> OpenCV statistical feature extraction
  -> Qwen2.5-VL domain observations
  -> GuardianEye weight router
  -> Selected HF, NTU, or RLVS checkpoint
  -> Violence classification
```

Qwen provides semantic observations. The router combines those observations
with measurable video statistics and selects the most suitable checkpoint.

## What Qwen Should Identify

Ask Qwen to identify domain-related information such as:

- Recording source: `cctv`, `broadcast`, `mobile`, `phone`, or `web video`.
- Environment: `ice rink`, `arena`, `street`, `indoor`, or `public space`.
- Camera style: `fixed`, `wide angle`, `tracking`, `handheld`, or `moving`.
- Clip structure: `five second window`, `continuous footage`,
  `short event clip`, `edited clip`, or `broadcast excerpt`.
- Domain indicators: hockey uniforms, ice rink, arena boards, security-camera
  timestamp, broadcast overlays, visible editing, or handheld camera movement.

Do not use generic violence descriptions such as `fight`, `violence`,
`assault`, or `attack` to select a dataset. These terms describe the
classification task and may occur in all three datasets.

## Do Not Ask Qwen To Select The Model

Avoid prompts such as:

```text
Which model should process this video: HF, NTU, or RLVS?
```

Qwen does not know the measured training distribution, checkpoint thresholds,
or router scoring logic. Instead, ask it to return structured observations that
the router can evaluate.

## Recommended Qwen Prompt

```text
Analyze this video only for its recording domain and visual style.

Do not determine whether the video is violent.
Do not select a machine-learning checkpoint.
Do not use violence-related actions as domain evidence.

Identify:
1. Recording source.
2. Environment.
3. Camera style.
4. Clip structure.
5. Visible domain indicators.
6. Confidence in the observations.

Return only valid JSON matching this structure:
{
  "overview": "short domain-focused description",
  "source": "cctv | broadcast | mobile | phone | web video | unknown",
  "environment": "short description",
  "camera": "fixed | wide angle | tracking | handheld | moving | unknown",
  "clip_structure": "five second window | continuous footage | short event clip | edited clip | broadcast excerpt | unknown",
  "domain_indicators": ["indicator"],
  "confidence": 0.0
}
```

## Example Qwen Output

```json
{
  "overview": "Continuous footage from a fixed security camera overlooking a public street.",
  "source": "cctv",
  "environment": "public street",
  "camera": "fixed",
  "clip_structure": "continuous footage",
  "domain_indicators": [
    "wide-angle view",
    "security-camera timestamp"
  ],
  "confidence": 0.88
}
```

## Pass Qwen Output To The Router

Validate and parse Qwen's JSON before passing it to the router:

```python
from model_weight_router import ModelWeightRouter

router = ModelWeightRouter(r"D:\RAG_Graduation_Translator\Model_Types")

qwen_output = {
    "overview": "Continuous footage from a fixed security camera overlooking a public street.",
    "source": "cctv",
    "environment": "public street",
    "camera": "fixed",
    "clip_structure": "continuous footage",
    "domain_indicators": [
        "wide-angle view",
        "security-camera timestamp",
    ],
    "confidence": 0.88,
}

specifications = {
    "source": qwen_output["source"],
    "environment": qwen_output["environment"],
    "camera": qwen_output["camera"],
    "clip_structure": qwen_output["clip_structure"],
}

decision = router.route_video(
    video_path,
    context=qwen_output["overview"],
    specifications=specifications,
)
```

The router returns:

- The selected `HF`, `NTU`, or `RLVS` dataset model.
- The selected checkpoint path.
- The correct model-specific decision threshold.
- Ranked domain-similarity estimates.
- Reasons for the routing decision.

## Confidence Handling

Qwen's confidence is not currently consumed directly by the router. Apply these
rules before using its observations:

- Confidence `>= 0.75`: use the context and specifications normally.
- Confidence between `0.50` and `0.75`: use only clearly visible observations.
- Confidence `< 0.50`: omit Qwen specifications and route mainly from OpenCV
  video statistics.

Never treat Qwen's confidence as violence confidence or routing confidence.

## Validate Qwen Output

Before routing:

- Require valid JSON.
- Reject unsupported specification values.
- Convert missing fields to `unknown` or omit them.
- Ensure confidence is a number between `0.0` and `1.0`.
- Ignore violence-action words as routing evidence.
- Do not allow Qwen to provide checkpoint paths or thresholds.

## Model Limitations

`Qwen2.5-VL-3B-Instruct-AWQ` is a relatively small quantized VLM. It may:

- Confuse CCTV, mobile, and web-sourced footage.
- Miss small security-camera timestamps or broadcast overlays.
- Produce inconsistent JSON without a strict prompt.
- Lose some visual-detail accuracy due to AWQ quantization.
- Describe video content correctly while misidentifying its source domain.

For this reason, always combine Qwen observations with the router's measurable
video statistics.

## Final Responsibility Boundaries

- **Qwen:** describes the video's domain and visual style.
- **Weight router:** selects the most suitable model checkpoint.
- **Selected HF/NTU/RLVS classifier:** determines violence confidence and verdict.
- **RAG/LLM layer:** explains the classifier result using retrieved context.

Qwen must not be treated as the final authority for checkpoint selection or the
violence verdict.

## Reference

Qwen2.5-VL supports native video understanding, temporal analysis, and
second-level event localization:

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
