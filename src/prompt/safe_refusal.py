"""Safe refusal prompt templates — trains conservative "I don't know" behavior.

In high-stakes industrial settings, a model that confidently gives
wrong advice is far more dangerous than one that refuses.  This module
provides prompt templates and few-shot examples that teach the LLM to:

1. Refuse when retrieved FMEA context is insufficient.
2. Never suggest actions outside its asset dictionary.
3. Always defer to human operators for irreversible actions (e.g.
   valve changes, shutdown decisions).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Few-shot examples that demonstrate safe refusal behavior.
# All values and tag names are fictitious.
SAFE_REFUSAL_EXAMPLES: List[Dict[str, str]] = [
    {
        "context": "RAG returned 0 results for tag FV-999. Topology: unknown.",
        "query": "What should we do about FV-999 high temperature alarm?",
        "good_response": (
            "Tag FV-999 was not found in the FMEA knowledge base or asset "
            "dictionary.  I cannot provide a diagnostic recommendation.  "
            "**Action: manual inspection required.**  Please verify the tag "
            "in the SCADA system and consult the shift supervisor."
        ),
        "bad_response": (
            "Try reducing the feed flow by 10% and check if the temperature "
            "drops.  This is a common fix for valve overheating."
        ),
    },
    {
        "context": (
            "FMEA entry #042: TE-301 bearing temperature high, possible "
            "cause: lubricant degradation.  Confidence: 0.45 (below threshold)."
        ),
        "query": "Is TE-301 about to fail? Should we shut down?",
        "good_response": (
            "FMEA entry #042 [Source: fmea_042] suggests possible lubricant "
            "degradation, but the confidence is only 0.45 (below the 0.6 "
            "operational threshold).  **I cannot recommend immediate shutdown "
            "based on this evidence alone.**  Suggested next step: dispatch "
            "a vibration analysis technician to TE-301 and obtain a lubricant "
            "sample for laboratory analysis before making a shutdown decision."
        ),
        "bad_response": (
            "Yes, shut down immediately.  The bearing is about to fail."
        ),
    },
    {
        "context": (
            "FMEA entry #018: PT-302 pressure drop across column T-301 "
            "exceeds 2.5 bar.  Possible flooding.  Severity=8.  Confidence=0.82."
        ),
        "query": "Column T-301 delta-P is 3.1 bar — what valve adjustment?",
        "good_response": (
            "FMEA #018 [Source: fmea_018] matches: flooding precursor with "
            "S=8, confidence=0.82.  The topology shows T-301 feeds T-302.  "
            "Recommended action: **gradually reduce feed valve FV-301 opening "
            "by 5% and increase reflux via FV-302 by 3%** while monitoring "
            "top pressure PT-301 for 10 minutes.  If ΔP does not decrease "
            "within 10 minutes, escalate to shift supervisor for possible "
            "controlled shutdown."
        ),
        "bad_response": (
            "Fully open FV-301 and close FV-302.  That should fix it."
        ),
    },
]

SAFE_REFUSAL_SYSTEM_PROMPT = (
    "You are a conservative industrial FMEA diagnostic agent.  "
    "Your primary directive: **do no harm.**\n\n"
    "Rules (strict — violations will be rejected by the guardrails gateway):\n"
    "1. ONLY use information from the provided FMEA context and topology.\n"
    "2. If confidence is below 0.6, explicitly state the uncertainty and "
    "recommend manual investigation — never guess.\n"
    "3. NEVER recommend irreversible actions (emergency shutdown, valve "
    "closure, venting) without explicit operator confirmation.\n"
    "4. When in doubt, respond: 'Insufficient evidence — manual inspection "
    "required.'\n"
    "5. Cite every factual claim with [Source: <id>].\n"
    "6. NEVER output tag names that are not in the provided asset dictionary.\n"
    "7. Abundance values MUST be in [0, 100] %.  Values outside this range "
    "indicate a sensor or data-processing error — flag it, don't diagnose it.\n"
)


class SafeRefusalPrompt:
    """Build prompts that encode industrial safety constraints.

    Usage::

        srp = SafeRefusalPrompt()
        system_prompt = srp.build_system_prompt()
        few_shot = srp.build_few_shot(n_examples=2)
    """

    def __init__(
        self,
        examples: List[Dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._examples = examples or SAFE_REFUSAL_EXAMPLES
        self._system_prompt = system_prompt or SAFE_REFUSAL_SYSTEM_PROMPT

    def build_system_prompt(self) -> str:
        """Return the safe-refusal system prompt."""
        return self._system_prompt

    def build_few_shot(self, n_examples: int = 3) -> str:
        """Return few-shot examples as a formatted string.

        Each example contains the "bad" response as a warning of what
        NOT to do, and the "good" response as the model to follow.
        """
        parts: List[str] = []
        for i, ex in enumerate(self._examples[:n_examples]):
            parts.append(f"### Example {i + 1}")
            parts.append(f"Context: {ex['context']}")
            parts.append(f"Query: {ex['query']}")
            parts.append(f"\n**CORRECT response:**\n{ex['good_response']}")
            parts.append(f"\n**WRONG response (hallucination — DO NOT DO THIS):**\n{ex['bad_response']}\n")
        return "\n".join(parts)

    def build_full_prompt(
        self,
        rag_context: str,
        alarm_tag: str,
        plc_values: Dict[str, float] | None = None,
        n_few_shot: int = 2,
    ) -> str:
        """Assemble the complete prompt: system + few-shot + context + query.

        Args:
            rag_context: concatenated RAG retrieval results.
            alarm_tag: the triggering PLC tag.
            plc_values: optional current readings.
            n_few_shot: number of few-shot examples to include.

        Returns:
            Complete prompt string ready for the LLM.
        """
        parts = [
            self._system_prompt,
            "\n---\n",
            self.build_few_shot(n_few_shot),
            "\n---\n",
            "## Current Situation\n",
            f"Alarm tag: {alarm_tag}\n",
        ]
        if plc_values:
            parts.append("Current PLC readings:\n")
            for tag, val in plc_values.items():
                parts.append(f"  {tag} = {val}\n")
        parts.append(f"\n## Retrieved FMEA Knowledge\n{rag_context}\n")
        parts.append("\n## Your Diagnostic Report (JSON format, with [Source:] citations)\n")
        return "".join(parts)


def build_safe_prompt(
    rag_context: str,
    alarm_tag: str,
    plc_values: Dict[str, float] | None = None,
) -> str:
    """Convenience: build a full safe-refusal prompt in one call."""
    return SafeRefusalPrompt().build_full_prompt(rag_context, alarm_tag, plc_values)
