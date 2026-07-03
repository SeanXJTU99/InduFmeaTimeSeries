"""SFT dataset builder — converts SOPs and FMEA entries into QA pairs.

Reads the declarative FMEA chunks (output of SemanticRewriter), the
distillation operating manuals, and historical fault tickets, and
produces instruction-following training examples::

    {"instruction": "<system prompt + alarm context>",
     "input": "<PLC readings and tag info>",
     "output": "<FMEA diagnosis with citations>"}

These pairs are used to fine-tune the base LLM via QLoRA.
All examples are fictitious.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SFTConfig:
    """SFT dataset builder parameters."""

    output_path: str = "data/sft_dataset.jsonl"
    instruction_template: str = (
        "You are an industrial FMEA diagnostic agent for a multi-stage "
        "cryogenic distillation system.  Based on the alarm signal and "
        "provided FMEA knowledge, diagnose the most likely failure mode.  "
        "Cite sources.  If uncertain, state 'Manual inspection required.'"
    )
    include_negative_examples: bool = True  # include "unknown" fallback cases
    max_examples: int = 5000


class SFTDatasetBuilder:
    """Build instruction-tuning datasets from FMEA knowledge.

    Usage::

        builder = SFTDatasetBuilder()
        builder.add_fmea_chunks(declarative_rows)
        builder.add_manual_qa(manual_qa_pairs)
        builder.save("data/sft_train.jsonl")
    """

    def __init__(self, config: SFTConfig | None = None) -> None:
        self._cfg = config or SFTConfig()
        self._examples: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fmea_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Convert declarative FMEA chunks into QA training pairs.

        Each chunk becomes one training example where the instruction is
        the system prompt, the input is the chunk's metadata context,
        and the output is the declarative sentence itself (teacher forcing
        the model to reproduce accurate FMEA knowledge).
        """
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            tag = meta.get("tag", "UNKNOWN")
            example = {
                "instruction": self._cfg.instruction_template,
                "input": (
                    f"Alarm: {tag} abnormal reading.  "
                    f"System: {meta.get('system', 'Unknown')}.  "
                    f"Retrieved FMEA context below."
                ),
                "output": chunk.get("page_content", ""),
            }
            self._examples.append(example)

    def add_manual_qa(self, qa_pairs: List[Dict[str, str]]) -> None:
        """Add pre-formatted QA pairs (e.g. from operating manuals).

        Args:
            qa_pairs: list of ``{"instruction": ..., "input": ..., "output": ...}``.
        """
        self._examples.extend(qa_pairs)

    def add_safe_refusal_example(self, tag: str, reason: str) -> None:
        """Add a training example that demonstrates safe refusal.

        Args:
            tag: the PLC tag for which no FMEA entry exists.
            reason: why the diagnosis cannot be made.
        """
        self._examples.append({
            "instruction": self._cfg.instruction_template,
            "input": (
                f"Alarm: {tag} abnormal reading.  "
                f"No FMEA entries found for this tag in the knowledge base."
            ),
            "output": (
                f"Tag {tag} was not found in the FMEA knowledge base.  "
                f"Reason: {reason}.  "
                f"**Action: manual inspection required.**  "
                f"Do NOT attempt to diagnose without verified FMEA data."
            ),
        })

    def build(self) -> List[Dict[str, str]]:
        """Return the collected training examples."""
        return self._examples[: self._cfg.max_examples]

    def save(self, path: str | None = None) -> None:
        """Write the dataset to a JSONL file.

        Args:
            path: output path (defaults to config value).
        """
        target = path or self._cfg.output_path
        with open(target, "w", encoding="utf-8") as f:
            for ex in self._examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_sft_dataset(
    fmea_chunks: List[Dict[str, Any]],
    manual_qa: List[Dict[str, str]] | None = None,
    output_path: str = "data/sft_dataset.jsonl",
) -> List[Dict[str, str]]:
    """Convenience: build and save an SFT dataset in one call."""
    builder = SFTDatasetBuilder(SFTConfig(output_path=output_path))
    builder.add_fmea_chunks(fmea_chunks)
    if manual_qa:
        builder.add_manual_qa(manual_qa)
    builder.add_safe_refusal_example("FV-999", "Tag not in asset dictionary")
    builder.save()
    return builder.build()
