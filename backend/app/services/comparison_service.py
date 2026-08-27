from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.services.multi_document_service import (
    multi_document_service,
)
from app.services.chat_service import (
    chat_service,
)


class ComparisonService:
    """
    ResearchGPT / PaperAxiom
    Academic Research Paper Comparison Service.

    Purpose
    -------
    Compare 2–10 selected research papers using:

        Selected papers
              ↓
        Balanced semantic retrieval
              ↓
        Evidence normalization
              ↓
        Cross-paper academic reasoning
              ↓
        LLM synthesis
              ↓
        Clean academic report

    Important design principles
    ----------------------------
    1. Every selected paper must be represented.
    2. Evidence is balanced across papers.
    3. The LLM performs synthesis rather than copying chunks.
    4. Paper-specific facts remain evidence-grounded.
    5. General academic knowledge may clarify implications.
    6. Missing evidence is explicitly acknowledged.
    7. The service should remain reasonably fast.
    8. Existing routes.py interface is preserved.
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MIN_PAPERS = 2
    MAX_PAPERS = 10

    DEFAULT_EVIDENCE_PER_PAPER = 8

    # Number of chunks retrieved from Qdrant per paper.
    RETRIEVAL_LIMIT_PER_PAPER = 8

    # Maximum chunks finally used per paper.
    MAX_EVIDENCE_PER_PAPER = 8

    # Maximum characters contributed by one paper.
    MAX_CHARS_PER_PAPER = 5000

    # IMPORTANT:
    # ChatService currently protects itself by limiting the prompt
    # size. We keep our final context comfortably below that limit.
    MAX_CONTEXT_CHARS = 10500

    # Much larger than the old 850-token comparison.
    LLM_MAX_TOKENS = 2200

    # ============================================================
    # RESEARCH DIMENSIONS
    # ============================================================

    COMPARISON_QUERY = """
    research problem objective aim motivation
    methodology research design experimental setup
    dataset data population participants patients samples
    preprocessing training validation testing
    model architecture algorithm deep learning machine learning
    neural network CNN transformer classifier segmentation detection
    results findings performance accuracy precision recall F1
    AUC Dice IoU sensitivity specificity evaluation
    baseline comparison contribution novelty innovation
    limitations weaknesses generalization robustness
    future work recommendations unresolved problems
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

    def compare(
        self,
        paper_ids: List[int],
        evidence_per_paper: int = DEFAULT_EVIDENCE_PER_PAPER,
        user_question: str = "",
    ) -> Dict[str, Any]:
        """
        Compare selected research papers.

        Existing route compatibility:

            comparison_service.compare(
                paper_ids=[...],
                evidence_per_paper=...
            )
        """

        # --------------------------------------------------------
        # Normalize IDs
        # --------------------------------------------------------

        normalized_ids = (
            self._normalize_ids(
                paper_ids
            )
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        if len(normalized_ids) < self.MIN_PAPERS:

            raise ValueError(
                "Paper comparison requires "
                "at least two different papers."
            )

        if len(normalized_ids) > self.MAX_PAPERS:

            raise ValueError(
                "Paper comparison supports a "
                "maximum of 10 papers."
            )

        # --------------------------------------------------------
        # Question
        # --------------------------------------------------------

        question = (
            str(user_question).strip()
            if user_question
            else ""
        )

        if not question:

            question = (
                "Compare the selected research papers "
                "academically. Explain their research "
                "objectives, methodology, datasets, models "
                "or approaches, results, contributions, "
                "limitations, similarities, differences, "
                "and implications for future research."
            )

        print(
            "=================================================="
        )

        print(
            "ResearchGPT | Paper Comparison"
        )

        print(
            f"Selected papers: {normalized_ids}"
        )

        print(
            f"Research question: {question}"
        )

        # --------------------------------------------------------
        # Retrieve balanced evidence
        # --------------------------------------------------------

        evidence = (
            self._collect_evidence(
                paper_ids=normalized_ids,
                evidence_per_paper=(
                    evidence_per_paper
                    or self.DEFAULT_EVIDENCE_PER_PAPER
                ),
                user_question=question,
            )
        )

        # --------------------------------------------------------
        # Build compact cross-paper context
        # --------------------------------------------------------

        evidence_context = (
            self._build_context(
                paper_ids=normalized_ids,
                evidence=evidence,
            )
        )

        # --------------------------------------------------------
        # Build academic prompt
        # --------------------------------------------------------

        prompt = (
            self._build_prompt(
                paper_ids=normalized_ids,
                evidence_context=evidence_context,
                question=question,
            )
        )

        # --------------------------------------------------------
        # LLM synthesis
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
        # Fallback
        # --------------------------------------------------------

        if not answer:

            generation_status = (
                "evidence_fallback"
            )

            answer = (
                self._build_fallback(
                    paper_ids=normalized_ids,
                    evidence=evidence,
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
                self._build_fallback(
                    paper_ids=normalized_ids,
                    evidence=evidence,
                )
            )

        # --------------------------------------------------------
        # Sources
        # --------------------------------------------------------

        sources = (
            self._build_sources(
                paper_ids=normalized_ids,
                evidence=evidence,
            )
        )

        # --------------------------------------------------------
        # Paper information
        # --------------------------------------------------------

        papers = []

        for paper_id in normalized_ids:

            paper_data = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            papers.append(
                {
                    "paper_id": paper_id,

                    "paper_name": (
                        paper_data.get(
                            "paper_name",
                            f"Paper {paper_id}",
                        )
                    ),
                }
            )

        # --------------------------------------------------------
        # Final result
        # --------------------------------------------------------

        result = {

            "paper_ids":
                normalized_ids,

            "papers_count":
                len(normalized_ids),

            "papers":
                papers,

            "question":
                question,

            "comparison":
                answer,

            "sources":
                sources,

            "generation_status":
                generation_status,

            "analysis_type":
                "academic_comparison",
        }

        print(
            "ResearchGPT | Paper Comparison "
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

        lines = []

        for line in text.split(
            "\n"
        ):

            cleaned = " ".join(
                line.split()
            ).strip()

            if cleaned:

                lines.append(
                    cleaned
                )

        return "\n".join(
            lines
        ).strip()

    # ============================================================
    # TEXT EXTRACTION
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

        keys = (
            "text",
            "content",
            "chunk",
            "page_content",
            "document",
            "body",
            "snippet",
        )

        for key in keys:

            value = item.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                text = (
                    self._clean_text(
                        value
                    )
                )

                if text:

                    return text

        return ""

    # ============================================================
    # PAPER ID EXTRACTION
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
    # PAPER NAME EXTRACTION
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
    # SCORE
    # ============================================================

    def _score(
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
    # DUPLICATE DETECTION
    # ============================================================

    def _is_duplicate(
        self,
        text: str,
        existing: List[str],
    ) -> bool:

        normalized = (
            " ".join(
                text.lower().split()
            )
        )

        if not normalized:

            return True

        for old in existing:

            old_normalized = (
                " ".join(
                    old.lower().split()
                )
            )

            if (
                normalized
                == old_normalized
            ):

                return True

            # Compare long chunks using word overlap.
            words_a = set(
                normalized.split()
            )

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
    # COLLECT EVIDENCE
    # ============================================================

    def _collect_evidence(
        self,
        paper_ids: List[int],
        evidence_per_paper: int,
        user_question: str,
    ) -> Dict[int, Dict[str, Any]]:

        evidence = {}

        # --------------------------------------------------------
        # Initialize every paper.
        # --------------------------------------------------------

        for paper_id in paper_ids:

            evidence[
                paper_id
            ] = {

                "paper_name":
                    f"Paper {paper_id}",

                "all_chunks":
                    [],
            }

        # --------------------------------------------------------
        # One broad semantic query.
        #
        # This is intentionally different from the old implementation
        # that made many sequential category searches.
        #
        # Fewer retrieval calls = faster and less failure-prone.
        # --------------------------------------------------------

        query = (
            f"{user_question} "
            f"{self.COMPARISON_QUERY}"
        )

        print(
            "ResearchGPT | Retrieving "
            "balanced comparison evidence..."
        )

        try:

            result = (
                self.multi_document_service.search(
                    query=query,
                    paper_ids=paper_ids,
                    limit_per_paper=max(
                        5,
                        min(
                            evidence_per_paper,
                            self.RETRIEVAL_LIMIT_PER_PAPER,
                        ),
                    ),
                )
            )

        except Exception as exc:

            print(
                "ResearchGPT | Comparison "
                f"retrieval failed: {exc}"
            )

            return evidence

        results = (
            self._normalize_results(
                result
            )
        )

        # --------------------------------------------------------
        # Put every result into the correct paper.
        # --------------------------------------------------------

        for item in results:

            if not isinstance(
                item,
                dict,
            ):

                continue

            paper_id = (
                self._extract_paper_id(
                    item
                )
            )

            if paper_id is None:

                continue

            if paper_id not in evidence:

                continue

            text = (
                self._extract_text(
                    item
                )
            )

            if not text:

                continue

            paper_name = (
                self._extract_paper_name(
                    item
                )
            )

            if paper_name:

                current_name = (
                    evidence[
                        paper_id
                    ].get(
                        "paper_name"
                    )
                )

                if (
                    not current_name
                    or current_name
                    == f"Paper {paper_id}"
                ):

                    evidence[
                        paper_id
                    ][
                        "paper_name"
                    ] = paper_name

            evidence[
                paper_id
            ][
                "all_chunks"
            ].append(
                {
                    "paper_id":
                        paper_id,

                    "paper_name":
                        (
                            paper_name
                            or
                            evidence[
                                paper_id
                            ][
                                "paper_name"
                            ]
                        ),

                    "text":
                        text,

                    "score":
                        self._score(
                            item
                        ),
                }
            )

        # --------------------------------------------------------
        # Balance evidence.
        # --------------------------------------------------------

        max_items = max(
            3,
            min(
                evidence_per_paper,
                self.MAX_EVIDENCE_PER_PAPER,
            ),
        )

        for paper_id in paper_ids:

            chunks = (
                evidence[
                    paper_id
                ][
                    "all_chunks"
                ]
            )

            # Highest relevance first.
            chunks.sort(
                key=self._score,
                reverse=True,
            )

            selected = []

            seen = []

            for chunk in chunks:

                text = chunk.get(
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

                # Limit individual chunk size.
                copied = dict(
                    chunk
                )

                copied[
                    "text"
                ] = text[:1600]

                selected.append(
                    copied
                )

                if len(
                    selected
                ) >= max_items:

                    break

            evidence[
                paper_id
            ][
                "all_chunks"
            ] = selected

            print(
                "ResearchGPT | "
                f"Paper {paper_id} | "
                f"{len(selected)} evidence items"
            )

        return evidence

    # ============================================================
    # NORMALIZE SEARCH RESULTS
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
    # BUILD LLM CONTEXT
    # ============================================================

    def _build_context(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        sections = []

        remaining = (
            self.MAX_CONTEXT_CHARS
        )

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            paper_data = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            paper_name = (
                paper_data.get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
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

            chunks = (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            )

            if not chunks:

                missing = (
                    "No relevant evidence "
                    "was retrieved for this paper.\n"
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

            for chunk_index, chunk in enumerate(
                chunks,
                start=1,
            ):

                text = (
                    self._clean_text(
                        chunk.get(
                            "text",
                            "",
                        )
                    )
                )

                if not text:

                    continue

                available_for_paper = min(
                    self.MAX_CHARS_PER_PAPER
                    - paper_chars,
                    remaining
                    - 100,
                )

                if available_for_paper <= 200:

                    break

                text = text[
                    :available_for_paper
                ]

                block = (
                    f"\nEvidence {chunk_index}:\n"
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
        )[:self.MAX_CONTEXT_CHARS]

    # ============================================================
    # BUILD PROMPT
    # ============================================================

    def _build_prompt(
        self,
        paper_ids: List[int],
        evidence_context: str,
        question: str,
    ) -> str:

        paper_labels = "\n".join(
            f"Paper {index}: ID {paper_id}"
            for index, paper_id
            in enumerate(
                paper_ids,
                start=1,
            )
        )

        return f"""
You are an expert academic research assistant
specializing in scientific literature analysis.

The researcher selected {len(paper_ids)} research papers.

============================================================
SELECTED PAPERS
============================================================

{paper_labels}

============================================================
RESEARCHER REQUEST
============================================================

{question}

============================================================
SOURCE MATERIAL
============================================================

{evidence_context}

============================================================
YOUR TASK
============================================================

Produce a high-quality academic comparison of the
selected research papers.

The goal is NOT to copy evidence.

The goal is to UNDERSTAND the studies and explain how
they relate to one another.

Use your own academic language and reasoning.

============================================================
IMPORTANT EVIDENCE RULES
============================================================

The supplied paper evidence is the primary source for
paper-specific facts.

You MAY use general academic knowledge to:

- explain terminology,
- clarify methodological concepts,
- explain why a methodological difference matters,
- interpret the academic significance of findings.

However, you MUST NOT use general knowledge to invent
paper-specific facts.

Never invent:

- authors,
- paper titles,
- datasets,
- sample sizes,
- patient populations,
- models,
- algorithms,
- numerical results,
- experiments,
- contributions,
- limitations,
- future work.

If a paper-specific detail is unavailable, state:

"Not identified in the available paper evidence."

============================================================
VERY IMPORTANT
============================================================

You MUST consider EVERY selected paper.

Do not accidentally compare only Paper 1 and Paper 2
when more papers are selected.

For two papers:
perform direct Paper 1 vs Paper 2 comparison.

For three or more papers:
identify common patterns, methodological clusters,
important differences, strongest findings, limitations,
and complementary approaches.

============================================================
ACADEMIC COMPARISON
============================================================

For a general comparison, use the following structure:

## Overall Comparison

Explain what the selected papers collectively investigate
and how their research directions relate.

## Research Objectives

Compare the research problems, aims, motivations,
research questions, or objectives.

## Methodology

Compare research design, experimental workflow,
preprocessing, training, validation, and methodology.

## Dataset and Data

Compare datasets, data sources, population,
sample characteristics, modalities, annotations,
and experimental splits when supported.

## Models and Approaches

Compare models, architectures, algorithms,
features, learning approaches, and technical methods.

## Results and Findings

Compare important reported findings and numerical
performance metrics.

Preserve numerical values accurately.

Explain what the results mean academically.

## Contributions

Explain what each study contributes and whether
the contributions are complementary or substantially
different.

## Limitations

Compare limitations explicitly supported by the papers.

Do not invent limitations.

## Key Similarities

Provide the strongest common themes across the papers.

## Key Differences

Provide the most important methodological,
dataset, objective, or result differences.

## Research Implications

Explain what the combined evidence suggests for
future research.

## Overall Assessment

Provide a concise final academic synthesis.

============================================================
WRITING STYLE
============================================================

Write like a graduate-level academic researcher.

The answer must be:

- analytical,
- precise,
- professional,
- evidence-grounded,
- readable,
- substantive.

Do not merely write:

"Paper 1 did X.
Paper 2 did Y."

Instead explain relationships.

Use transitions such as:

"Similarly,"
"In contrast,"
"Compared with,"
"However,"
"Collectively,"
"Taken together,"
"An important distinction is,"
"These differences suggest"

only when appropriate.

Do not repeat the same fact in multiple sections.

Do not mention:

- prompts,
- embeddings,
- Qdrant,
- retrieval,
- chunks,
- internal processing,
- system instructions,
- model selection.

Do not expose hidden reasoning.

Return ONLY the final academic comparison.
"""

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
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

        # --------------------------------------------------------
        # First attempt
        # --------------------------------------------------------

        try:

            response = method(
                prompt,
                max_tokens=self.LLM_MAX_TOKENS,
            )

            if isinstance(
                response,
                str,
            ):

                cleaned = (
                    response.strip()
                )

                if cleaned:

                    return cleaned

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

            # Compatibility with older ChatService.
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
                    f"LLM fallback failed: {exc}"
                )

        except Exception as exc:

            print(
                "ResearchGPT | "
                f"LLM comparison failed: {exc}"
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

        # Remove markdown code fences.
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

        # Remove accidental internal reasoning tags.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove obvious internal prefixes.
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

        # Remove excessive blank lines.
        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # FALLBACK
    # ============================================================

    def _build_fallback(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        sections = [
            "# Paper Comparison",
            "",
            (
                "The comparison below is based on the "
                "available evidence from the selected papers."
            ),
            "",
        ]

        # --------------------------------------------------------
        # Individual paper evidence
        # --------------------------------------------------------

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            paper_data = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            paper_name = (
                paper_data.get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
            )

            sections.extend(
                [
                    f"## Paper {index}: {paper_name}",
                    "",
                ]
            )

            chunks = (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            )

            if not chunks:

                sections.extend(
                    [
                        (
                            "No relevant evidence was "
                            "retrieved for this paper."
                        ),
                        "",
                    ]
                )

                continue

            for chunk in chunks[:5]:

                text = (
                    self._clean_text(
                        chunk.get(
                            "text",
                            "",
                        )
                    )
                )

                if text:

                    sections.extend(
                        [
                            text[:1200],
                            "",
                        ]
                    )

        # --------------------------------------------------------
        # Comparative note
        # --------------------------------------------------------

        sections.extend(
            [
                "## Comparative Assessment",
                "",
                (
                    "The available evidence provides "
                    "paper-specific information for the "
                    "selected studies. A definitive comparison "
                    "should be based on the reported objectives, "
                    "methods, datasets, results, contributions, "
                    "and limitations without introducing "
                    "unsupported claims."
                ),
            ]
        )

        return "\n".join(
            sections
        )

    # ============================================================
    # BUILD SOURCES
    # ============================================================

    def _build_sources(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        sources = []

        for paper_id in paper_ids:

            paper_data = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            paper_name = (
                paper_data.get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
            )

            for chunk in (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            ):

                sources.append(
                    {
                        "paper_id":
                            paper_id,

                        "paper_name":
                            paper_name,

                        "text":
                            chunk.get(
                                "text",
                                "",
                            ),

                        "score":
                            chunk.get(
                                "score",
                                0.0,
                            ),
                    }
                )

        return sources


# ================================================================
# SERVICE INSTANCE
# ================================================================

comparison_service = ComparisonService()