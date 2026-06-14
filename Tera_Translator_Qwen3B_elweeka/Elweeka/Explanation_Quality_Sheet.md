# Guardian Eye RAG - Explanation Quality Sheet

| Example ID | Input Verdict | Confidence | Packet Summary | References | Narrative / Answer | Status | Pass/Fail Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `explain_pass_with_frames` | `violence` | `0.91` | Evidence describes two people pushing near a chair. | `ref-violence-contact` | Deterministic fallback narrative preserves violence and confidence, using packet, references, and frame refs only. | Pass | Guardrails pass; no invented weapons, people, timing, or confidence changes. |
| `explain_text_only_fallback` | `non-violence` | `0.74` | Evidence describes normal activity without violent contact. | none | Text-only fallback narrative preserves non-violence and confidence. | Pass | Frames are absent, fallback limitation is stated, no unsupported details are added. |
| `explain_regenerate_required` | `violence` | `0.94` | Evidence describes rapid motion between two people. | none | Mock unsafe candidate says no violence occurred and changes confidence to 0.80. | Fail / Regenerate | Guardrails reject verdict contradiction and confidence modification; regeneration instruction is returned. |
| `ask_historical_stored_records` | stored records | mixed | Stored incident packets include one knife incident and one bottle incident. | stored records only | Historical answer returns only the matching stored knife incident. | Pass | Metadata filters apply before ranking; summary uses stored packet and narrative only. |

## Quality Criteria

- Verdict is preserved.
- Confidence is preserved.
- Narrative is 2-4 grounded sentences for explanations.
- No unsupported people, weapons, objects, actions, exact timing, or blame.
- Limitations are stated when frames, references, or historical detail are incomplete.
- Historical answers use stored packet summaries and stored narratives only.

## Result

The saved Day 5 mock/demo examples pass the Elweeka quality checks.
