from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.services.multi_document_service import (
    multi_document_service,
)

from app.services.chat_service import (
    chat_service,
)


class LiteratureReviewService:
    """
    ResearchGPT / ResearchGPT
    Evidence-grounded academic literature review service.

    Pipeline
    --------
    Selected papers
        ↓
    Balanced semantic retrieval
        ↓
    Evidence normalization
        ↓
    Compact academic context
        ↓
    LLM literature synthesis
        ↓
    Clean academic review
        ↓
    Evidence fallback if LLM unavailable

    Design goals
    ------------
    - Reliable
    - Evidence-grounded
    - Balanced across selected papers
    - Reasonably fast
    - Compatible with existing routes
    - Suitable for graduate-level academic writing
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MIN_PAPERS = 1
    MAX_PAPERS = 10

    DEFAULT_EVIDENCE_PER_PAPER = 8

    RETRIEVAL_LIMIT_PER_PAPER = 8

    MAX_EVIDENCE_PER_PAPER = 8

    MAX_CHARS_PER_PAPER = 5000

    # Keep total prompt comfortably below ChatService's
    # internal prompt-size protection.
    MAX_CONTEXT_CHARS = 10500

    # Literature reviews need more generation space than
    # simple question answering.
    LLM_MAX_TOKENS = 2200

    # ============================================================
    # RETRIEVAL QUERY
    # ============================================================

    LITERATURE_QUERY = """
    research problem research question objective aim
    motivation significance theoretical background
    methodology research design framework workflow pipeline
    experimental setup preprocessing training validation testing
    dataset data source population participants patients samples
    images videos annotations labels data split
    model architecture algorithm deep learning machine learning
    neural network CNN transformer classification segmentation
    detection feature extraction optimization
    results findings evaluation performance accuracy precision
    recall F1 AUC Dice IoU sensitivity specificity
    comparison baseline ablation experiment
    contribution novelty innovation advancement
    limitations weaknesses challenges generalization robustness
    bias computational limitations clinical limitations
    future work recommendations extensions unresolved problems
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self) -> None:

        self.multi_document_service = (
            multi_document_service
        )

        self.chat_service = (
            chat_service
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate(
        self,
        paper_ids: List[int],
    ) -> Dict[str, Any]:
        """
        Generate an evidence-grounded literature review.

        Existing backend contract is preserved:

        {
            "paper_ids": [...],
            "papers_count": ...,
            "review": "...",
            "sources": [...],
            "source_count": ...,
            "generation_status": "..."
        }
        """

        # --------------------------------------------------------
        # Normalize paper IDs
        # --------------------------------------------------------

        unique_paper_ids = (
            self._normalize_ids(
                paper_ids
            )
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        if not unique_paper_ids:

            raise ValueError(
                "At least one valid paper is required "
                "for a literature review."
            )

        if (
            len(unique_paper_ids)
            > self.MAX_PAPERS
        ):

            raise ValueError(
                "You can select a maximum of 10 papers."
            )

        print(
            "=================================================="
        )

        print(
            "ResearchGPT | Literature Review"
        )

        print(
            f"Selected papers: {unique_paper_ids}"
        )

        # --------------------------------------------------------
        # Retrieve balanced evidence
        # --------------------------------------------------------

        evidence_by_paper = (
            self._retrieve_evidence(
                unique_paper_ids
            )
        )

        # --------------------------------------------------------
        # Build compact context
        # --------------------------------------------------------

        evidence_context = (
            self._build_context(
                paper_ids=unique_paper_ids,
                evidence_by_paper=evidence_by_paper,
            )
        )

        # --------------------------------------------------------
        # Build academic prompt
        # --------------------------------------------------------

        prompt = (
            self._build_prompt(
                paper_ids=unique_paper_ids,
                evidence_context=evidence_context,
            )
        )

        # --------------------------------------------------------
        # Generate review
        # --------------------------------------------------------

        answer = (
            self._call_llm(
                prompt
            )
        )

        generation_status = (
            "ai_generated"
        )

        # --------------------------------------------------------
        # Retry with compact prompt
        # --------------------------------------------------------

        if not answer:

            print(
                "ResearchGPT | Literature review "
                "generation failed. Retrying..."
            )

            compact_prompt = (
                prompt
                + """

IMPORTANT:
Produce a more concise version of the literature
review while preserving the most important:

- research trends
- methodological differences
- datasets
- important findings
- contributions
- limitations
- research gaps

Do not omit important numerical results when
they are explicitly supported.
"""
            )

            answer = (
                self._call_llm(
                    compact_prompt,
                    max_tokens=1500,
                )
            )

        # --------------------------------------------------------
        # Evidence fallback
        # --------------------------------------------------------

        if not answer:

            print(
                "ResearchGPT | LLM unavailable. "
                "Using evidence-grounded fallback."
            )

            generation_status = (
                "evidence_fallback"
            )

            answer = (
                self._build_fallback_review(
                    paper_ids=unique_paper_ids,
                    evidence_by_paper=evidence_by_paper,
                )
            )

        # --------------------------------------------------------
        # Clean
        # --------------------------------------------------------

        answer = (
            self._clean_output(
                answer
            )
        )

        if not answer:

            generation_status = (
                "evidence_fallback"
            )

            answer = (
                self._build_fallback_review(
                    paper_ids=unique_paper_ids,
                    evidence_by_paper=evidence_by_paper,
                )
            )

        # --------------------------------------------------------
        # Sources
        # --------------------------------------------------------

        sources = (
            self._build_sources(
                paper_ids=unique_paper_ids,
                evidence_by_paper=evidence_by_paper,
            )
        )

        # --------------------------------------------------------
        # Final response
        # --------------------------------------------------------

        result = {

            "paper_ids":
                unique_paper_ids,

            "papers_count":
                len(unique_paper_ids),

            "review":
                answer,

            "sources":
                sources,

            "source_count":
                len(sources),

            "generation_status":
                generation_status,
        }

        print(
            "ResearchGPT | Literature Review "
            f"completed | {generation_status}"
        )

        print(
            "=================================================="
        )

        return result

    # ============================================================
    # NORMALIZE IDS
    # ============================================================

    def _normalize_ids(
        self,
        paper_ids: List[int],
    ) -> List[int]:

        normalized = []

        for value in paper_ids or []:

            try:

                paper_id = int(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if (
                paper_id > 0
                and paper_id not in normalized
            ):

                normalized.append(
                    paper_id
                )

        return normalized

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    def _clean_text(
        self,
        value: Any,
    ) -> str:

        if value is None:

            return ""

        text = str(
            value
        )

        text = (
            text
            .replace(
                "\x00",
                " ",
            )
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        # Preserve readable paragraph boundaries.
        paragraphs = []

        for paragraph in text.split(
            "\n"
        ):

            cleaned = " ".join(
                paragraph.split()
            ).strip()

            if cleaned:

                paragraphs.append(
                    cleaned
                )

        return "\n".join(
            paragraphs
        ).strip()

    # ============================================================
    # NORMALIZE TEXT
    # ============================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        return (
            " ".join(
                text.lower().split()
            )
        )

    # ============================================================
    # DUPLICATE DETECTION
    # ============================================================

    def _is_duplicate(
        self,
        text: str,
        existing: List[str],
    ) -> bool:

        normalized = (
            self._normalize_text(
                text
            )
        )

        if not normalized:

            return True

        words_a = set(
            normalized.split()
        )

        for old in existing:

            old_normalized = (
                self._normalize_text(
                    old
                )
            )

            if (
                normalized
                == old_normalized
            ):

                return True

            words_b = set(
                old_normalized.split()
            )

            if (
                len(words_a) < 25
                or len(words_b) < 25
            ):

                continue

            union = (
                words_a
                | words_b
            )

            if not union:

                continue

            similarity = (
                len(
                    words_a
                    & words_b
                )
                / len(union)
            )

            if similarity >= 0.88:

                return True

        return False

    # ============================================================
    # EXTRACT PAPER ID
    # ============================================================

    def _extract_paper_id(
        self,
        item: Dict[str, Any],
    ) -> Optional[int]:

        if not isinstance(
            item,
            dict,
        ):

            return None

        possible_keys = (
            "paper_id",
            "paperId",
            "document_id",
            "documentId",
        )

        containers = [
            item,
            item.get(
                "metadata"
            ),
            item.get(
                "payload"
            ),
            item.get(
                "meta"
            ),
        ]

        for container in containers:

            if not isinstance(
                container,
                dict,
            ):

                continue

            for key in possible_keys:

                value = (
                    container.get(
                        key
                    )
                )

                if value is None:

                    continue

                try:

                    return int(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

        return None

    # ============================================================
    # EXTRACT PAPER NAME
    # ============================================================

    def _extract_paper_name(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:

        if not isinstance(
            item,
            dict,
        ):

            return None

        possible_keys = (
            "paper_name",
            "paper_title",
            "title",
            "document_title",
            "filename",
            "file_name",
            "name",
        )

        containers = [
            item,
            item.get(
                "metadata"
            ),
            item.get(
                "payload"
            ),
            item.get(
                "meta"
            ),
        ]

        for container in containers:

            if not isinstance(
                container,
                dict,
            ):

                continue

            for key in possible_keys:

                value = (
                    container.get(
                        key
                    )
                )

                if isinstance(
                    value,
                    str,
                ):

                    value = (
                        value.strip()
                    )

                    if value:

                        return value

        return None

    # ============================================================
    # EXTRACT TEXT
    # ============================================================

    def _extract_text(
        self,
        item: Dict[str, Any],
    ) -> str:

        if not isinstance(
            item,
            dict,
        ):

            return ""

        possible_keys = (
            "text",
            "content",
            "chunk",
            "page_content",
            "document",
            "body",
            "snippet",
        )

        for key in possible_keys:

            value = item.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                cleaned = (
                    self._clean_text(
                        value
                    )
                )

                if cleaned:

                    return cleaned

        return ""

    # ============================================================
    # SCORE
    # ============================================================

    def _get_score(
        self,
        item: Dict[str, Any],
    ) -> float:

        if not isinstance(
            item,
            dict,
        ):

            return 0.0

        for key in (
            "score",
            "similarity",
            "relevance_score",
            "rerank_score",
        ):

            value = item.get(
                key
            )

            try:

                return float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        return 0.0

    # ============================================================
    # NORMALIZE SEARCH RESULT
    # ============================================================

    def _normalize_results(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:

        if result is None:

            return []

        if isinstance(
            result,
            list,
        ):

            return [
                item
                for item in result
                if isinstance(
                    item,
                    dict,
                )
            ]

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "results",
                "sources",
                "documents",
                "chunks",
                "evidence",
                "data",
            ):

                value = result.get(
                    key
                )

                if isinstance(
                    value,
                    list,
                ):

                    return [
                        item
                        for item in value
                        if isinstance(
                            item,
                            dict,
                        )
                    ]

            if self._extract_text(
                result
            ):

                return [
                    result
                ]

        return []

    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
    ) -> Dict[int, List[Dict[str, Any]]]:

        evidence_by_paper = {
            paper_id: []
            for paper_id in paper_ids
        }

        print(
            "ResearchGPT | Retrieving "
            "balanced literature evidence..."
        )

        query = (
            self.LITERATURE_QUERY
        )

        try:

            result = (
                self.multi_document_service.search(
                    query=query,
                    paper_ids=paper_ids,
                    limit_per_paper=(
                        self.RETRIEVAL_LIMIT_PER_PAPER
                    ),
                )
            )

        except Exception as exc:

            print(
                "ResearchGPT | Literature "
                f"retrieval failed: {exc}"
            )

            return evidence_by_paper

        results = (
            self._normalize_results(
                result
            )
        )

        # --------------------------------------------------------
        # Assign results to papers
        # --------------------------------------------------------

        for item in results:

            paper_id = (
                self._extract_paper_id(
                    item
                )
            )

            if paper_id is None:

                continue

            if paper_id not in (
                evidence_by_paper
            ):

                continue

            text = (
                self._extract_text(
                    item
                )
            )

            if not text:

                continue

            existing = [
                x.get(
                    "text",
                    "",
                )
                for x in evidence_by_paper[
                    paper_id
                ]
            ]

            if self._is_duplicate(
                text,
                existing,
            ):

                continue

            paper_name = (
                self._extract_paper_name(
                    item
                )
            )

            evidence_by_paper[
                paper_id
            ].append(
                {
                    "paper_id":
                        paper_id,

                    "paper_name":
                        (
                            paper_name
                            or
                            f"Paper {paper_id}"
                        ),

                    "text":
                        text,

                    "score":
                        self._get_score(
                            item
                        ),
                }
            )

        # --------------------------------------------------------
        # Balance and rank
        # --------------------------------------------------------

        for paper_id in paper_ids:

            items = (
                evidence_by_paper[
                    paper_id
                ]
            )

            items.sort(
                key=self._get_score,
                reverse=True,
            )

            selected = []

            seen = []

            for item in items:

                text = item.get(
                    "text",
                    "",
                )

                if not text:

                    continue

                if self._is_duplicate(
                    text,
                    seen,
                ):

                    continue

                seen.append(
                    text
                )

                selected.append(
                    {
                        **item,
                        "text":
                            text[:1600],
                    }
                )

                if len(
                    selected
                ) >= self.MAX_EVIDENCE_PER_PAPER:

                    break

            evidence_by_paper[
                paper_id
            ] = selected

            print(
                "ResearchGPT | "
                f"Paper {paper_id} | "
                f"{len(selected)} evidence items"
            )

        return evidence_by_paper

    # ============================================================
    # BUILD CONTEXT
    # ============================================================

    def _build_context(
        self,
        paper_ids: List[int],
        evidence_by_paper: Dict[
            int,
            List[Dict[str, Any]],
        ],
    ) -> str:

        sections = []

        remaining = (
            self.MAX_CONTEXT_CHARS
        )

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            items = (
                evidence_by_paper.get(
                    paper_id,
                    [],
                )
            )

            paper_name = (
                items[0].get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
                if items
                else f"Paper {paper_id}"
            )

            header = (
                "\n"
                + "=" * 65
                + "\n"
                + f"PAPER {index}\n"
                + f"Paper ID: {paper_id}\n"
                + f"Paper Name: {paper_name}\n"
                + "=" * 65
                + "\n"
            )

            if len(header) >= remaining:

                break

            sections.append(
                header
            )

            remaining -= len(
                header
            )

            if not items:

                missing = (
                    "No relevant paper evidence "
                    "was identified.\n"
                )

                if len(missing) < remaining:

                    sections.append(
                        missing
                    )

                    remaining -= len(
                        missing
                    )

                continue

            paper_chars = 0

            for item_index, item in enumerate(
                items,
                start=1,
            ):

                text = (
                    self._clean_text(
                        item.get(
                            "text",
                            "",
                        )
                    )
                )

                if not text:

                    continue

                available = min(
                    self.MAX_CHARS_PER_PAPER
                    - paper_chars,
                    remaining
                    - 120,
                )

                if available <= 200:

                    break

                text = text[
                    :available
                ]

                block = (
                    f"\nEvidence {item_index}:\n"
                    f"{text}\n"
                )

                sections.append(
                    block
                )

                block_length = len(
                    block
                )

                paper_chars += (
                    block_length
                )

                remaining -= (
                    block_length
                )

                if (
                    paper_chars
                    >= self.MAX_CHARS_PER_PAPER
                ):

                    break

            if remaining <= 300:

                break

        return "".join(
            sections
        )[
            :self.MAX_CONTEXT_CHARS
        ]

    # ============================================================
    # BUILD PROMPT
    # ============================================================

    def _build_prompt(
        self,
        paper_ids: List[int],
        evidence_context: str,
    ) -> str:

        paper_list = "\n".join(
            f"- Paper {index}: ID {paper_id}"
            for index, paper_id
            in enumerate(
                paper_ids,
                start=1,
            )
        )

        return f"""
You are an expert academic research assistant
specializing in systematic literature synthesis.

You have been given evidence from
{len(paper_ids)} selected research papers.

Your task is to write a coherent,
graduate-level academic literature review.

============================================================
SELECTED PAPERS
============================================================

{paper_list}

============================================================
PAPER MATERIAL
============================================================

{evidence_context}

============================================================
PRIMARY OBJECTIVE
============================================================

Synthesize the selected studies into a literature review.

Do NOT simply produce a paper-by-paper summary.

Instead, identify:

- common research themes,
- differences in research objectives,
- methodological trends,
- datasets and data characteristics,
- model and algorithm trends,
- major findings,
- complementary approaches,
- limitations,
- unresolved issues,
- evidence-supported research gaps,
- future research directions.

The reader should be able to understand the
research landscape represented by the selected
papers after reading the review.

============================================================
EVIDENCE RULES
============================================================

The supplied paper material is the factual basis
for paper-specific claims.

You MUST NOT invent:

- authors,
- publication years,
- datasets,
- sample sizes,
- patient numbers,
- model names,
- algorithms,
- metrics,
- numerical results,
- experimental settings,
- contributions,
- limitations,
- future work.

If a requested detail is not available, write:

"Not identified in the available paper evidence."

You may use general academic knowledge only to
explain terminology or clarify why a reported
methodological difference is meaningful.

Do not use general knowledge to fabricate
paper-specific information.

============================================================
REQUIRED STRUCTURE
============================================================

# Literature Review

Start with a strong introductory synthesis that
describes the research area represented by the
selected studies.

Do not begin with a generic definition of the field.

## 1. Research Focus and Objectives

Explain what research problems the studies address.

Synthesize the objectives across papers and highlight
important differences.

Avoid repetitive paper-by-paper descriptions.

## 2. Methodological Approaches

Compare the major methodological approaches.

Discuss:

- research design,
- experimental workflow,
- preprocessing,
- training,
- validation,
- testing,
- methodological frameworks,
- important architectural differences.

Focus on meaningful methodological trends.

## 3. Datasets and Experimental Data

Synthesize the datasets and data sources used.

Mention:

- dataset names,
- data modalities,
- sample characteristics,
- population,
- annotations,
- train/validation/test splits,

ONLY when supported by the supplied material.

Explain how differences in data may affect
interpretation of results when such an interpretation
is academically justified.

## 4. Models and Computational Techniques

Discuss important models, algorithms, architectures,
and computational techniques.

Do not create a long model inventory.

Focus on techniques that matter to the studies'
research contributions.

## 5. Results and Research Findings

Synthesize the major findings across studies.

Preserve explicit numerical metrics accurately.

For example, if the evidence reports:

- accuracy,
- precision,
- recall,
- F1,
- AUC,
- Dice,
- IoU,
- sensitivity,
- specificity,

retain those values accurately.

Do not invent missing metrics.

Explain important differences in reported outcomes
when the evidence supports such interpretation.

## 6. Contributions and Novelty

Explain what the studies contribute to the field.

Discuss whether the contributions:

- complement one another,
- address different aspects of the problem,
- improve existing approaches,
- introduce methodological innovation.

Do not claim novelty unless supported by the paper material.

## 7. Limitations and Future Directions

Discuss limitations explicitly reported by the studies.

Then synthesize explicitly reported future directions.

Do not invent limitations simply because a method
might theoretically have one.

## 8. Comparative Synthesis

This section is extremely important.

Move beyond paper-by-paper description.

Explain:

- major similarities,
- important differences,
- methodological trends,
- data differences,
- differences in objectives,
- differences in findings,
- complementary strengths,
- unresolved methodological issues.

Use connected academic paragraphs.

## 9. Research Gap

Identify a research gap ONLY if it can be reasonably
supported by the supplied literature.

A research gap may arise from:

- consistent limitations across studies,
- an underexplored dataset or population,
- a methodological limitation,
- insufficient generalization,
- conflicting findings,
- missing evaluation,
- an unresolved research problem.

If the evidence does not support a reliable gap,
write:

"An evidence-supported research gap could not be
established from the available paper material."

Do not manufacture a gap.

## 10. Overall Literature Assessment

Conclude with a concise but substantive synthesis
of the research landscape represented by the selected
papers.

Explain what the combined literature suggests
for future research.

============================================================
ACADEMIC WRITING REQUIREMENTS
============================================================

Use professional academic English.

The review should sound like a section of a
graduate-level research thesis or academic paper.

Use:

- connected paragraphs,
- analytical transitions,
- precise terminology,
- comparative reasoning,
- evidence-grounded claims.

Avoid:

- repetitive summaries,
- excessive bullet lists,
- unnecessary tables,
- generic textbook explanations,
- unsupported claims.

Useful transitions include:

"Similarly,"
"In contrast,"
"Compared with,"
"However,"
"Collectively,"
"Taken together,"
"These findings suggest,"
"An important distinction is,"
"Across the selected studies"

Use them naturally rather than mechanically.

============================================================
DO NOT MENTION INTERNAL SYSTEM DETAILS
============================================================

Do not mention:

- LLM,
- language model,
- prompt,
- embeddings,
- Qdrant,
- retrieval,
- chunks,
- vector database,
- internal processing,
- system instructions,
- similarity scores.

Do not expose hidden reasoning.

Return ONLY the final academic literature review.
"""

    # ============================================================
    # CALL LLM
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:

        method = getattr(
            self.chat_service,
            "_call_llm",
            None,
        )

        if not callable(
            method
        ):

            print(
                "ResearchGPT | "
                "ChatService._call_llm unavailable."
            )

            return None

        tokens = (
            max_tokens
            or self.LLM_MAX_TOKENS
        )

        # --------------------------------------------------------
        # Normal call
        # --------------------------------------------------------

        try:

            response = method(
                prompt,
                max_tokens=tokens,
            )

            if isinstance(
                response,
                str,
            ):

                response = (
                    response.strip()
                )

                if response:

                    return response

            if isinstance(
                response,
                dict,
            ):

                for key in (
                    "answer",
                    "response",
                    "content",
                    "text",
                    "message",
                ):

                    value = response.get(
                        key
                    )

                    if isinstance(
                        value,
                        str,
                    ):

                        value = (
                            value.strip()
                        )

                        if value:

                            return value

        except TypeError:

            # Compatibility with older
            # ChatService implementations.
            try:

                response = method(
                    prompt
                )

                if isinstance(
                    response,
                    str,
                ):

                    return (
                        response.strip()
                    )

            except Exception as exc:

                print(
                    "ResearchGPT | "
                    f"LLM compatibility call failed: {exc}"
                )

        except Exception as exc:

            print(
                "ResearchGPT | "
                f"Literature generation failed: {exc}"
            )

        return None

    # ============================================================
    # CLEAN OUTPUT
    # ============================================================

    def _clean_output(
        self,
        text: Any,
    ) -> str:

        if text is None:

            return ""

        text = str(
            text
        ).strip()

        # Remove accidental markdown code fences.
        text = re.sub(
            r"^```(?:markdown|md|text)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        # Remove hidden reasoning tags if returned
        # by a model.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        # Remove accidental assistant prefixes.
        prefixes = [
            "final answer:",
            "final response:",
            "assistant:",
        ]

        for prefix in prefixes:

            if text.lower().startswith(
                prefix
            ):

                text = (
                    text[
                        len(prefix):
                    ]
                    .strip()
                )

        # Normalize excessive blank lines.
        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # FALLBACK REVIEW
    # ============================================================

    def _build_fallback_review(
        self,
        paper_ids: List[int],
        evidence_by_paper: Dict[
            int,
            List[Dict[str, Any]],
        ],
    ) -> str:

        sections = []

        sections.append(
            "# Literature Review\n\n"
        )

        sections.append(
            (
                "The selected research papers address "
                "the research themes represented by the "
                "available paper material. The following "
                "synthesis is restricted to information "
                "supported by the available material.\n"
            )
        )

        # --------------------------------------------------------
        # Research focus
        # --------------------------------------------------------

        sections.append(
            "\n## 1. Research Focus and Objectives\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "objective",
                "research",
                "problem",
                "aim",
                "motivation",
                "purpose",
            ],
            max_items=2,
        )

        # --------------------------------------------------------
        # Methodology
        # --------------------------------------------------------

        sections.append(
            "\n## 2. Methodological Approaches\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "method",
                "methodology",
                "model",
                "architecture",
                "framework",
                "approach",
                "pipeline",
                "experiment",
            ],
            max_items=2,
        )

        # --------------------------------------------------------
        # Dataset
        # --------------------------------------------------------

        sections.append(
            "\n## 3. Datasets and Experimental Data\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "dataset",
                "data",
                "sample",
                "patient",
                "population",
                "images",
                "annotations",
            ],
            max_items=2,
        )

        # --------------------------------------------------------
        # Models
        # --------------------------------------------------------

        sections.append(
            "\n## 4. Models and Computational Techniques\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "model",
                "network",
                "cnn",
                "transformer",
                "algorithm",
                "deep learning",
                "machine learning",
                "architecture",
            ],
            max_items=2,
        )

        # --------------------------------------------------------
        # Results
        # --------------------------------------------------------

        sections.append(
            "\n## 5. Results and Research Findings\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "result",
                "finding",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "dice",
                "iou",
                "sensitivity",
                "specificity",
                "performance",
            ],
            max_items=2,
        )

        # --------------------------------------------------------
        # Contributions
        # --------------------------------------------------------

        sections.append(
            "\n## 6. Contributions and Novelty\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "contribution",
                "novel",
                "novelty",
                "innovation",
                "proposed",
                "advancement",
            ],
            max_items=2,
        )

        # --------------------------------------------------------
        # Limitations
        # --------------------------------------------------------

        sections.append(
            "\n## 7. Limitations and Future Directions\n\n"
        )

        self._append_category_fallback(
            sections=sections,
            paper_ids=paper_ids,
            evidence_by_paper=evidence_by_paper,
            category_keywords=[
                "limitation",
                "weakness",
                "constraint",
                "challenge",
                "future",
                "recommendation",
                "extension",
            ],
            max_items=3,
        )

        # --------------------------------------------------------
        # Comparative synthesis
        # --------------------------------------------------------

        sections.append(
            "\n## 8. Comparative Synthesis\n\n"
        )

        sections.append(
            (
                "The available material provides evidence "
                "concerning the objectives, methods, datasets, "
                "models, and findings represented by the "
                "selected studies. A definitive cross-study "
                "comparison should be restricted to similarities "
                "and differences directly supported by the "
                "available material."
            )
        )

        # --------------------------------------------------------
        # Research gap
        # --------------------------------------------------------

        sections.append(
            "\n\n## 9. Research Gap\n\n"
        )

        sections.append(
            (
                "An evidence-supported research gap could "
                "not be established from the available "
                "paper material alone. Further examination "
                "of the complete studies may be required "
                "to establish a reliable research gap."
            )
        )

        # --------------------------------------------------------
        # Overall assessment
        # --------------------------------------------------------

        sections.append(
            "\n\n## 10. Overall Literature Assessment\n\n"
        )

        sections.append(
            (
                "Overall, the selected papers provide "
                "evidence concerning their respective "
                "research objectives, methodological "
                "approaches, datasets, computational "
                "techniques, findings, contributions, "
                "limitations, and future directions. "
                "The combined material provides a structured "
                "overview of the represented research area, "
                "while conclusions beyond the available "
                "paper-specific information should be "
                "treated cautiously."
            )
        )

        return "".join(
            sections
        )

    # ============================================================
    # FALLBACK CATEGORY HELPER
    # ============================================================

    def _append_category_fallback(
        self,
        sections: List[str],
        paper_ids: List[int],
        evidence_by_paper: Dict[
            int,
            List[Dict[str, Any]],
        ],
        category_keywords: List[str],
        max_items: int = 2,
    ) -> None:

        for paper_id in paper_ids:

            items = (
                evidence_by_paper.get(
                    paper_id,
                    [],
                )
            )

            matched = []

            for item in items:

                text = (
                    self._clean_text(
                        item.get(
                            "text",
                            "",
                        )
                    )
                )

                lower = (
                    text.lower()
                )

                if any(
                    keyword.lower()
                    in lower
                    for keyword
                    in category_keywords
                ):

                    matched.append(
                        item
                    )

                if len(
                    matched
                ) >= max_items:

                    break

            if not matched:

                sections.append(
                    f"Paper {paper_id}: "
                    "Not identified in the available "
                    "paper evidence.\n\n"
                )

                continue

            sections.append(
                f"Paper {paper_id}:\n"
            )

            for item in matched:

                sections.append(
                    f"- {item.get('text', '')}\n"
                )

            sections.append(
                "\n"
            )

    # ============================================================
    # BUILD SOURCES
    # ============================================================

    def _build_sources(
        self,
        paper_ids: List[int],
        evidence_by_paper: Dict[
            int,
            List[Dict[str, Any]],
        ],
    ) -> List[Dict[str, Any]]:

        sources = []

        for paper_id in paper_ids:

            for item in (
                evidence_by_paper.get(
                    paper_id,
                    [],
                )
            ):

                sources.append(
                    {
                        "paper_id":
                            paper_id,

                        "paper_name":
                            item.get(
                                "paper_name",
                                f"Paper {paper_id}",
                            ),

                        "text":
                            item.get(
                                "text",
                                "",
                            ),

                        "score":
                            item.get(
                                "score",
                                0.0,
                            ),
                    }
                )

        return sources


# ================================================================
# SERVICE INSTANCE
# ================================================================

literature_review_service = (
    LiteratureReviewService()
)