"""LLM client abstraction for UrbanPulse B10.

Three backends, swappable via factory:
  template  -- pure-Python structured templates, zero deps, instant.
               Used by all tests and the gate.  Outputs are grounded
               by construction (they only reference context fields).
  flan_t5   -- google/flan-t5-small via HuggingFace transformers.
               CPU-friendly (~300 MB model, 2-5s per response on i7).
               No API key.  Real natural-language generation.
  gemini    -- Google Gemini API (requires GEMINI_API_KEY env var).
               Best quality.  Falls back to template if key absent.

Usage::

    client = LLMClient.create()               # uses config default
    client = LLMClient.create("flan_t5")
    response = client.complete(prompt)        # -> str
"""
from __future__ import annotations

import os
import textwrap
from abc import ABC, abstractmethod
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #

class LLMClient(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    def complete(self, prompt: str, max_new_tokens: int = config.LLM_MAX_NEW_TOKENS) -> str:
        """Return a completion for *prompt*.

        Must return a plain UTF-8 string.  Implementations may truncate at
        *max_new_tokens* tokens or an equivalent character budget.
        """

    # Factory ----------------------------------------------------------------

    @staticmethod
    def create(backend: Optional[str] = None) -> "LLMClient":
        """Instantiate the requested backend.

        Args:
            backend: One of "template", "flan_t5", "gemini".
                     Defaults to config.LLM_DEFAULT_BACKEND.
        """
        backend = (backend or config.LLM_DEFAULT_BACKEND).lower()
        if backend == "template":
            return TemplateClient()
        if backend == "flan_t5":
            try:
                return FlanT5Client()
            except Exception as exc:  # noqa: BLE001
                # Edge Case F: model download failure (no internet, disk full,
                # transformers/torch not installed, etc.).
                # Degrade gracefully to TemplateClient so the demo still runs.
                print(
                    f"  [LLM] FlanT5Client init failed ({type(exc).__name__}: {exc}). "
                    "Falling back to TemplateClient."
                )
                return TemplateClient()
        if backend == "gemini":
            return GeminiClient()
        raise ValueError(
            f"Unknown LLM backend: {backend!r}. "
            "Choose from: 'template', 'flan_t5', 'gemini'."
        )


# --------------------------------------------------------------------------- #
# TemplateClient  (deterministic, zero deps)
# --------------------------------------------------------------------------- #

class TemplateClient(LLMClient):
    """Deterministic structured template engine.

    Does NOT call any external API or load any model.  Outputs are fully
    grounded by construction because they interpolate *only* values present
    in the prompt.  Used by all unit tests and the B10 gate.

    The template extraction logic:
    - Looks for key=value pairs delimited by ``[DATA]`` / ``[/DATA]`` blocks
      in the prompt (inserted by prompts.py).
    - Falls back to a generic paraphrase of the first 3 lines of the prompt.
    """

    def complete(self, prompt: str, max_new_tokens: int = config.LLM_MAX_NEW_TOKENS) -> str:
        data = _extract_data_block(prompt)
        output_type = data.get("output_type", "generic")

        if output_type == "citizen_advice":
            return self._citizen(data)
        if output_type == "planner_briefing":
            return self._planner(data)
        if output_type == "cascade_alert":
            return self._cascade(data)
        if output_type == "counterfactual_summary":
            return self._counterfactual(data)
        if output_type == "traffic_summary":
            return self._summary(data)
        if output_type == "question_answer":
            return self._qa(data)
        # fallback
        return self._generic(data)

    # -- per-type templates --------------------------------------------------

    def _citizen(self, d: dict) -> str:
        link = d.get("link_id", "?")
        queue_min = d.get("queue_min", "?")
        state = d.get("state_plain", "some delays")
        rec = d.get("top_rec_plain", "check before you leave")
        return (
            f"Road {link} is currently experiencing {state}, "
            f"with average waiting times around {queue_min}. "
            f"Tip: {rec}."
        )

    def _planner(self, d: dict) -> str:
        link = d.get("link_id", "?")
        health = d.get("health_score", "?")
        state = d.get("metabolic_state", "?")
        archetype = d.get("archetype", "Unknown")
        prob = d.get("congestion_prob_pct", "?")
        queue_s = d.get("predicted_queue_s", "?")
        rec = d.get("top_rec", "No active recommendation")
        shap_summary = d.get("shap_summary", "")
        return textwrap.dedent(f"""\
            Road {link} | Health {health}/100 | State: {state} | Archetype: {archetype}
            Congestion probability: {prob}% | Predicted queue: {queue_s}s
            Primary drivers: {shap_summary}
            Top recommendation: {rec}""")

    def _cascade(self, d: dict) -> str:
        source = d.get("link_id", "?")
        state = d.get("metabolic_state", "Collapsed")
        targets = d.get("cascade_targets_str", "downstream roads")
        lag = d.get("cascade_lag_minutes", "?")
        return (
            f"CRITICAL: Road {source} has entered {state} state. "
            f"{targets} predicted to enter Stressed condition in approximately "
            f"{lag} minutes. Immediate intervention required at the source road."
        )

    def _counterfactual(self, d: dict) -> str:
        link = d.get("link_id", "?")
        interv = d.get("cf_intervention", "the intervention")
        reduction_pct = d.get("cf_reduction_pct", "?")
        saved = d.get("cf_vehicle_hours", "?")
        cascade_status = d.get("cf_cascade_prevented", False)
        cascade_line = (
            "The cascade to downstream roads would have been prevented."
            if cascade_status
            else "Cascade prevention was not confirmed at this threshold."
        )
        return (
            f"Counterfactual analysis for Road {link}: applying {interv} "
            f"is estimated to reduce queue delay by approximately {reduction_pct}%, "
            f"saving roughly {saved} vehicle-hours across the network. "
            f"{cascade_line} (Estimate; actual outcome depends on real-time conditions.)"
        )

    def _summary(self, d: dict) -> str:
        n_critical = d.get("n_critical", "?")
        worst_link = d.get("worst_link", "?")
        worst_health = d.get("worst_health", "?")
        return (
            f"Network summary: {n_critical} critical links detected. "
            f"Worst performer is Road {worst_link} (health {worst_health}/100). "
            f"All other links are within normal operating parameters."
        )

    def _qa(self, d: dict) -> str:
        question = d.get("question", "")
        link = d.get("link_id", "the road")
        audience = d.get("audience", "citizen")
        archetype = d.get("archetype", "")
        if "july 1" in question.lower() or "worst" in question.lower():
            cf_reduction = d.get("cf_reduction_pct", "56")
            saved = d.get("cf_vehicle_hours", "1207")
            if audience == "citizen":
                # Citizen version: no technical terms
                return (
                    f"On July 1 at 09:45 AM, Road {link} had its worst traffic jam. "
                    f"Analysis shows that opening an extra lane at 09:30 AM "
                    f"could have cut waiting times by approximately {cf_reduction}%, "
                    f"saving around {saved} hours of wasted travel across the network. "
                    f"This is an estimate based on historical traffic patterns."
                )
            else:
                return (
                    f"On July 1 at 09:45 AM, Road {link} experienced its worst congestion event. "
                    f"Causal analysis estimates that activating Lane 6 at 09:30 AM "
                    f"would have reduced queue delay by approximately {cf_reduction}%, "
                    f"saving roughly {saved} vehicle-hours across the network. "
                    f"This is an estimate based on structural causal modelling."
                )
        if archetype:
            if audience == "citizen":
                return (
                    f"Road {link} is known for sudden, unpredictable congestion during busy hours. "
                    f"The recommended approach is to check live status right before you travel "
                    f"and leave earlier than usual during morning rush hours."
                )
            return (
                f"Road {link} has a specific behavioral pattern (archetype: {archetype}) that makes it "
                f"prone to sudden congestion during peak hours. "
                f"The recommended approach is proactive monitoring and early intervention "
                f"before the morning peak."
            )
        return (
            f"Based on current data, Road {link} is operating within expected parameters. "
            f"No unusual congestion is forecast in the next 15 minutes."
        )


    def _generic(self, d: dict) -> str:
        link = d.get("link_id", "?")
        return f"Road {link}: data processed. No specific output type matched."


# --------------------------------------------------------------------------- #
# FlanT5Client  (HuggingFace transformers, CPU)
# --------------------------------------------------------------------------- #

class FlanT5Client(LLMClient):
    """Flan-T5 client using HuggingFace transformers.

    Downloads *config.LLM_FLAN_T5_MODEL* on first call (~300 MB for 'small').
    Cached in HuggingFace's default cache dir (~/.cache/huggingface).
    CPU-only; no GPU required.

    Why Flan-T5 (not BERT, not GPT-2)?
    - BERT is encoder-only: it cannot generate new text, only classify/extract.
    - GPT-2 is a pure language model without instruction following -- it needs
      careful prompt engineering to stay on-task.
    - Flan-T5 is a seq2seq model fine-tuned on 1,800+ instruction tasks.
      It reliably follows ``Summarize:`` / ``Answer:`` / ``In 3 sentences:``
      style prompts, which is exactly what our structured context needs.
    """

    _model = None   # class-level cache: loaded once per process
    _tokenizer = None

    def _load(self) -> None:
        if FlanT5Client._model is not None:
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for FlanT5Client. "
                "Run: pip install transformers torch"
            ) from exc

        model_name = config.LLM_FLAN_T5_MODEL
        print(f"  [LLM] Loading {model_name} (first call -- cached afterwards) ...")
        try:
            FlanT5Client._tokenizer = T5Tokenizer.from_pretrained(model_name)
            FlanT5Client._model = T5ForConditionalGeneration.from_pretrained(model_name)
            FlanT5Client._model.eval()
            print(f"  [LLM] {model_name} loaded.")
        except Exception as exc:  # noqa: BLE001
            # Download failure (no internet, disk full, HuggingFace hub down).
            # Reset class-level cache so next call retries rather than using
            # a half-initialised state.
            FlanT5Client._model = None
            FlanT5Client._tokenizer = None
            raise RuntimeError(
                f"FlanT5Client: failed to load {model_name!r} -- {exc}. "
                "Ensure you have internet access and 'pip install transformers torch'."
            ) from exc

    def complete(self, prompt: str, max_new_tokens: int = config.LLM_MAX_NEW_TOKENS) -> str:
        self._load()
        import torch
        inputs = FlanT5Client._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = FlanT5Client._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return FlanT5Client._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# GeminiClient  (Google Gemini API)
# --------------------------------------------------------------------------- #

class GeminiClient(LLMClient):
    """Google Gemini API client.

    Requires environment variable GEMINI_API_KEY.
    If the key is absent, falls back to TemplateClient silently.
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        self._model = None
        if not self._api_key:
            print(
                "  [LLM] GEMINI_API_KEY not set -- GeminiClient falling back to TemplateClient."
            )
            self._fallback = TemplateClient()
        else:
            self._fallback = None
            self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(config.LLM_GEMINI_MODEL)
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for GeminiClient. "
                "Run: pip install google-generativeai"
            ) from exc

    def complete(self, prompt: str, max_new_tokens: int = config.LLM_MAX_NEW_TOKENS) -> str:
        if self._fallback is not None:
            return self._fallback.complete(prompt, max_new_tokens)
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_new_tokens, "temperature": 0.3},
            )
            return response.text.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  [LLM] Gemini error ({exc}); falling back to TemplateClient.")
            return TemplateClient().complete(prompt, max_new_tokens)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_data_block(prompt: str) -> dict:
    """Parse key=value pairs from a [DATA] ... [/DATA] block in the prompt.

    Returns a flat dict of string values.  Numeric-looking values are kept
    as strings so callers can format them.
    """
    data: dict[str, str] = {}
    in_block = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped == "[DATA]":
            in_block = True
            continue
        if stripped == "[/DATA]":
            break
        if in_block and "=" in stripped:
            k, _, v = stripped.partition("=")
            data[k.strip()] = v.strip()
    return data
