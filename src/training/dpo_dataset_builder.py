"""DPO dataset builder — constructs Chosen/Rejected preference pairs.

After SFT teaches the model domain knowledge, DPO (Direct Preference
Optimization) aligns its behaviour: the model learns to prefer cautious,
well-cited FMEA diagnoses (Chosen) over confident-but-ungrounded
speculation (Rejected).

Each training example is a triple:
    (prompt, chosen_response, rejected_response)

All examples are fictitious industrial safety scenarios.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DPOConfig:
    """DPO dataset builder parameters."""

    output_path: str = "data/dpo_dataset.jsonl"
    min_confidence_for_chosen: float = 0.70
    max_examples: int = 2000
    # Rejection reasons injected into the rejected response for variety
    rejection_templates: List[str] = (
        "Try adjusting {tag} by 10% — this usually fixes the issue.",
        "Based on general engineering knowledge, you should {action}.",
        "I am confident this is {cause}. No need to verify.",
        "The problem is almost certainly {cause}. Proceed with {action} immediately.",
    )


class DPODatasetBuilder:
    """Build DPO preference pairs for industrial safety alignment.

    For each FMEA chunk, a **Chosen** response is the verified,
    source-cited diagnosis.  A **Rejected** response is synthetically
    generated to exhibit unsafe behaviour: uncited claims, overconfidence,
    or suggestions to change valve positions without verification.

    Usage::

        builder = DPODatasetBuilder()
        builder.add_from_fmea(chunks)
        builder.add_safety_refusal_pairs(safety_examples)
        builder.save()
    """

    def __init__(self, config: DPOConfig | None = None) -> None:
        self._cfg = config or DPOConfig()
        self._triples: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_from_fmea(self, chunks: List[Dict[str, Any]]) -> None:
        """Build DPO pairs from FMEA chunks.

        Chosen = the correct FMEA diagnosis with citation.
        Rejected = a plausible-sounding but unsafe guess.
        """
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            tag = meta.get("tag", "N/A")
            prompt = (
                f"Alarm: {tag} reading abnormal.  "
                f"System: {meta.get('system', 'Unknown')}.  "
                f"Retrieved FMEA context provided below.  "
                f"What is the diagnosis?"
            )

            chosen = (
                f"[Source: {meta.get('source', chunk.get('source_id', 'fmea_XXX'))}] "
                f"{chunk.get('page_content', '')}\n\n"
                f"**Recommended action:** {meta.get('recommended_action', 'Manual inspection.')}\n"
                f"Confidence: {meta.get('confidence', 0.75):.2f}.  "
                f"If uncertain, manual inspection is required."
            )

            rejected = (
                f"I think the issue is likely related to {tag} malfunction.  "
                f"Try reducing the feed flow by 15% and see if the alarm clears.  "
                f"This is a common fix in distillation columns.  "
                f"(Note: no FMEA source cited — this is general speculation.)"
            )

            self._triples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    def add_safety_refusal_pair(
        self, tag: str, scenario: str, correct_refusal: str, unsafe_guess: str
    ) -> None:
        """Add a manually crafted safety refusal DPO pair.

        Args:
            tag: fictitious PLC tag.
            scenario: description of the alarm scenario.
            correct_refusal: the safe "I don't know — inspect manually" response.
            unsafe_guess: the dangerous speculative response.
        """
        self._triples.append({
            "prompt": f"Alarm: {tag} abnormal.  Scenario: {scenario}.  Diagnosis?",
            "chosen": correct_refusal,
            "rejected": unsafe_guess,
        })

    def build(self) -> List[Dict[str, str]]:
        """Return the collected DPO triples."""
        return self._triples[: self._cfg.max_examples]

    def save(self, path: str | None = None) -> None:
        """Write triples to a JSONL file for TRL DPOTrainer.

        Args:
            path: output path.
        """
        target = path or self._cfg.output_path
        with open(target, "w", encoding="utf-8") as f:
            for triple in self._triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")


def build_dpo_dataset(
    fmea_chunks: List[Dict[str, Any]],
    output_path: str = "data/dpo_dataset.jsonl",
) -> List[Dict[str, str]]:
    """Convenience: build and save a DPO dataset."""
    builder = DPODatasetBuilder(DPOConfig(output_path=output_path))
    builder.add_from_fmea(fmea_chunks)
    # Add canonical safety-refusal examples
    builder.add_safety_refusal_pair(
        tag="FV-999",
        scenario="Unknown valve tag — no FMEA or topology entry exists.",
        correct_refusal=(
            "Tag FV-999 is not in the asset dictionary or FMEA knowledge base.  "
            "I cannot provide a diagnosis.  **Manual inspection required.**  "
            "Do NOT adjust any valves without verified FMEA guidance."
        ),
        unsafe_guess=(
            "FV-999 is probably a control valve.  Try opening it to 50% "
            "and monitor for 10 minutes.  This is standard practice."
        ),
    )
    builder.save()
    return builder.build()
