from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.services.multi_document_service import (
    multi_document_service,
)

from app.services.chat_service import (
    chat_service,
)


class ResearchGapService:
    """
    ResearchGPT Research Gap Analysis Service.

    Purpose
    -------
    Analyze selected research papers and identify genuine,
    evidence-supported research gaps.

    Architecture
    ------------
        Selected Papers
              ↓
        Broad Evidence Retrieval
              ↓
        Evidence Cleaning / Deduplication
              ↓
        Paper-wise Organization
              ↓
        Cross-paper LLM Synthesis
              ↓
        Structured Research Gap Report

    Important design principles
    ---------------------------
    1. Do not treat missing retrieved evidence as proof
       that something does not exist in the original paper.

    2. Do not invent research gaps.

    3. Distinguish:
       - reported limitation
       - reported future work
       - evidence-supported inference
       - proposed research direction

    4. Compare papers against one another.

    5. Preserve the existing backend response structure.
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MAX_PAPERS = 10

    RETRIEVAL_LIMIT_PER_PAPER = 10

    MAX_EVIDENCE_PER_PAPER = 12

    MAX_CHARS_PER_EVIDENCE = 1800

    MAX_CHARS_PER_PAPER = 6500

    MAX_TOTAL_CONTEXT_CHARS = 22000

    FINAL_SYNTHESIS_MAX_TOKENS = 4200

    RETRY_MAX_TOKENS = 2500

    # ============================================================
    # RESEARCH DIMENSIONS
    # ============================================================

    CATEGORIES = [
        "Research Problem and Objective",
        "Background and Motivation",
        "Methodology and Experimental Design",
        "Data, Dataset and Population",
        "Models, Algorithms and Techniques",
        "Results and Evaluation",
        "Comparison with Existing Methods",
        "Contribution and Novelty",
        "Limitations and Failure Cases",
        "Future Work and Recommendations",
    ]

    CATEGORY_KEYWORDS = {

        "Research Problem and Objective": [
            "research problem",
            "research question",
            "objective",
            "aim",
            "purpose",
            "problem",
            "challenge",
            "motivation",
        ],

        "Background and Motivation": [
            "background",
            "motivation",
            "clinical",
            "scientific",
            "importance",
            "significance",
            "need",
            "context",
        ],

        "Methodology and Experimental Design": [
            "method",
            "methodology",
            "framework",
            "workflow",
            "pipeline",
            "experiment",
            "training",
            "testing",
            "validation",
            "preprocessing",
        ],

        "Data, Dataset and Population": [
            "dataset",
            "data",
            "patient",
            "patients",
            "population",
            "sample",
            "samples",
            "annotation",
            "annotations",
            "images",
            "videos",
            "records",
        ],

        "Models, Algorithms and Techniques": [
            "model",
            "models",
            "algorithm",
            "architecture",
            "network",
            "cnn",
            "transformer",
            "deep learning",
            "machine learning",
            "classification",
            "segmentation",
            "detection",
        ],

        "Results and Evaluation": [
            "result",
            "results",
            "finding",
            "findings",
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
            "evaluation",
        ],

        "Comparison with Existing Methods": [
            "baseline",
            "comparison",
            "compared",
            "existing method",
            "state of the art",
            "benchmark",
            "ablation",
            "previous work",
        ],

        "Contribution and Novelty": [
            "contribution",
            "novel",
            "novelty",
            "innovation",
            "proposed",
            "new approach",
            "advancement",
            "improvement",
        ],

        "Limitations and Failure Cases": [
            "limitation",
            "limitations",
            "weakness",
            "weaknesses",
            "failure",
            "constraint",
            "generalization",
            "robustness",
            "bias",
            "challenge",
            "unable",
            "difficult",
        ],

        "Future Work and Recommendations": [
            "future work",
            "future research",
            "future",
            "recommendation",
            "recommendations",
            "extension",
            "extend",
            "improve",
            "next step",
            "research opportunity",
        ],
    }

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
        Generate a complete research-gap analysis.

        Existing API is preserved.
        """

        normalized_ids = (
            self._normalize_paper_ids(
                paper_ids
            )
        )

        # --------------------------------------------------------
        # No papers
        # --------------------------------------------------------

        if not normalized_ids:

            return {
                "research_gap": (
                    "No papers were selected for "
                    "research-gap analysis."
                ),
                "papers_count": 0,
                "source_count": 0,
                "sources": [],
                "paper_analyses": {},
                "evidence": {},
            }

        # --------------------------------------------------------
        # Maximum papers
        # --------------------------------------------------------

        normalized_ids = (
            normalized_ids[
                :self.MAX_PAPERS
            ]
        )

        print(
            "=================================================="
        )

        print(
            "ResearchGPT | Research Gap Analysis"
        )

        print(
            f"Selected papers: {normalized_ids}"
        )

        # --------------------------------------------------------
        # STEP 1
        # --------------------------------------------------------

        evidence = (
            self._retrieve_evidence(
                normalized_ids
            )
        )

        # --------------------------------------------------------
        # STEP 2
        # --------------------------------------------------------

        sources = (
            self._flatten_sources(
                evidence
            )
        )

        # --------------------------------------------------------
        # STEP 3
        # --------------------------------------------------------

        paper_analyses = (
            self._build_paper_understandings(
                normalized_ids,
                evidence,
            )
        )

        # --------------------------------------------------------
        # STEP 4
        # --------------------------------------------------------

        research_gap = (
            self._generate_cross_paper_synthesis(
                paper_ids=normalized_ids,
                paper_analyses=paper_analyses,
                evidence=evidence,
            )
        )

        # --------------------------------------------------------
        # Retry if necessary
        # --------------------------------------------------------

        if not research_gap:

            print(
                "ResearchGPT | Retrying research-gap synthesis..."
            )

            research_gap = (
                self._generate_retry_synthesis(
                    paper_ids=normalized_ids,
                    evidence=evidence,
                )
            )

        # --------------------------------------------------------
        # Fallback
        # --------------------------------------------------------

        if not research_gap:

            print(
                "ResearchGPT | Using evidence-grounded "
                "research-gap fallback."
            )

            research_gap = (
                self._fallback_report(
                    paper_ids=normalized_ids,
                    paper_analyses=paper_analyses,
                    evidence=evidence,
                )
            )

        # --------------------------------------------------------
        # Clean output
        # --------------------------------------------------------

        research_gap = (
            self._clean_text_output(
                research_gap
            )
        )

        if not research_gap:

            research_gap = (
                self._fallback_report(
                    paper_ids=normalized_ids,
                    paper_analyses=paper_analyses,
                    evidence=evidence,
                )
            )

        print(
            "ResearchGPT | Research Gap Analysis completed."
        )

        print(
            f"Evidence sources: {len(sources)}"
        )

        print(
            "=================================================="
        )

        return {
            "research_gap": research_gap,

            "papers_count":
                len(normalized_ids),

            "source_count":
                len(sources),

            "sources":
                sources,

            "paper_analyses":
                paper_analyses,

            "evidence":
                evidence,
        }

    # ============================================================
    # PAPER ID NORMALIZATION
    # ============================================================

    def _normalize_paper_ids(
        self,
        paper_ids: List[int],
    ) -> List[int]:

        if not paper_ids:

            return []

        normalized = []

        for paper_id in paper_ids:

            try:

                value = int(
                    paper_id
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if value <= 0:

                continue

            if value not in normalized:

                normalized.append(
                    value
                )

        return normalized

    # ============================================================
    # BROAD RETRIEVAL QUERY
    # ============================================================

    def _build_research_gap_query(
        self,
    ) -> str:

        return """
        research problem research question objective aim purpose
        motivation background significance research challenge

        methodology research design framework workflow pipeline
        preprocessing training validation testing experiment

        dataset data source population patients participants
        sample size images videos annotations labels
        train validation test split

        models algorithms architectures deep learning
        machine learning CNN transformer classification
        segmentation detection feature extraction optimization

        results findings evaluation performance accuracy
        precision recall F1 AUC Dice IoU sensitivity specificity
        statistical significance baseline comparison

        contribution novelty innovation proposed method
        improvement advancement

        limitation limitations weakness weaknesses failure cases
        generalization robustness bias dataset limitations
        clinical limitations computational limitations

        future work future research recommendations extensions
        unresolved issues research opportunities

        state of the art previous studies existing methods
        gaps unanswered questions incomplete validation
        cross-dataset evaluation external validation
        clinical applicability practical deployment
        """
    
    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:

        evidence = {}

        query = (
            self._build_research_gap_query()
        )

        print(
            "ResearchGPT | Performing broad evidence retrieval..."
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

        except TypeError:

            # Compatibility with older service signature.
            try:

                result = (
                    self.multi_document_service.search(
                        query=query,
                        paper_ids=paper_ids,
                        limit=(
                            self.RETRIEVAL_LIMIT_PER_PAPER
                        ),
                    )
                )

            except Exception as exc:

                print(
                    "ResearchGPT | Retrieval failed: "
                    f"{exc}"
                )

                result = []

        except Exception as exc:

            print(
                "ResearchGPT | Retrieval failed: "
                f"{exc}"
            )

            result = []

        results = (
            self._normalize_results(
                result
            )
        )

        # --------------------------------------------------------
        # Initialize paper structures
        # --------------------------------------------------------

        for paper_id in paper_ids:

            evidence[paper_id] = {
                "paper_name":
                    f"Paper {paper_id}",

                "categories": {
                    category: []
                    for category
                    in self.CATEGORIES
                },

                "all_chunks": [],
            }

        # --------------------------------------------------------
        # Process results
        # --------------------------------------------------------

        for item in results:

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
                self._detect_paper_name(
                    item
                )
            )

            if paper_name:

                evidence[
                    paper_id
                ][
                    "paper_name"
                ] = paper_name

            copied = dict(
                item
            )

            copied[
                "paper_id"
            ] = paper_id

            copied[
                "text"
            ] = text

            copied[
                "score"
            ] = self._result_score(
                item
            )

            # ----------------------------------------------------
            # Avoid duplicates
            # ----------------------------------------------------

            existing_texts = [
                self._extract_text(
                    x
                )
                for x in evidence[
                    paper_id
                ][
                    "all_chunks"
                ]
            ]

            if self._is_duplicate(
                text,
                existing_texts,
            ):

                continue

            evidence[
                paper_id
            ][
                "all_chunks"
            ].append(
                copied
            )

        # --------------------------------------------------------
        # Rank and categorize locally
        # --------------------------------------------------------

        for paper_id in paper_ids:

            chunks = (
                evidence[
                    paper_id
                ][
                    "all_chunks"
                ]
            )

            chunks.sort(
                key=self._result_score,
                reverse=True,
            )

            chunks = chunks[
                :self.MAX_EVIDENCE_PER_PAPER
            ]

            evidence[
                paper_id
            ][
                "all_chunks"
            ] = chunks

            categories = (
                evidence[
                    paper_id
                ][
                    "categories"
                ]
            )

            for chunk in chunks:

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                best_category = (
                    self._classify_evidence(
                        text
                    )
                )

                if best_category:

                    categories[
                        best_category
                    ].append(
                        chunk
                    )

            print(
                "ResearchGPT | "
                f"Paper {paper_id} | "
                f"{len(chunks)} evidence items"
            )

        return evidence

    # ============================================================
    # RESULT NORMALIZATION
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
                "chunks",
                "documents",
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

                if isinstance(
                    value,
                    dict,
                ):

                    nested = (
                        self._normalize_results(
                            value
                        )
                    )

                    if nested:

                        return nested

            if self._extract_text(
                result
            ):

                return [
                    result
                ]

        return []

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

        keys = (
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

            for key in keys:

                value = container.get(
                    key
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

    def _detect_paper_name(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:

        if not isinstance(
            item,
            dict,
        ):

            return None

        keys = (
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

            for key in keys:

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

                cleaned = (
                    self._clean_source_text(
                        value
                    )
                )

                if cleaned:

                    return cleaned

        return ""

    # ============================================================
    # CLEAN SOURCE TEXT
    # ============================================================

    def _clean_source_text(
        self,
        text: str,
    ) -> str:

        if not text:

            return ""

        text = (
            str(text)
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
    # SCORE
    # ============================================================

    def _result_score(
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

        # Some systems expose distance instead.
        distance = item.get(
            "distance"
        )

        try:

            return -float(
                distance
            )

        except (
            TypeError,
            ValueError,
        ):

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
            self._normalize_for_comparison(
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
                self._normalize_for_comparison(
                    old
                )
            )

            if not old_normalized:

                continue

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
    # NORMALIZE FOR COMPARISON
    # ============================================================

    def _normalize_for_comparison(
        self,
        text: str,
    ) -> str:

        return (
            " ".join(
                str(text)
                .lower()
                .split()
            )
        )

    # ============================================================
    # CLASSIFY EVIDENCE
    # ============================================================

    def _classify_evidence(
        self,
        text: str,
    ) -> str:

        lower = (
            text.lower()
        )

        best_category = (
            "Research Problem and Objective"
        )

        best_score = 0

        for category, keywords in (
            self.CATEGORY_KEYWORDS.items()
        ):

            score = 0

            for keyword in keywords:

                if keyword.lower() in lower:

                    score += 1

                    # Strong weighting for explicit
                    # limitation/future language.
                    if category in (
                        "Limitations and Failure Cases",
                        "Future Work and Recommendations",
                    ):

                        score += 1

            if score > best_score:

                best_score = score

                best_category = category

        return best_category

    # ============================================================
    # BUILD PAPER UNDERSTANDINGS
    # ============================================================

    def _build_paper_understandings(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> Dict[int, str]:

        analyses = {}

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

            chunks = (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            )

            if not chunks:

                analyses[
                    paper_id
                ] = (
                    f"Paper {paper_id} "
                    f"({paper_name}): "
                    "No relevant evidence was retrieved."
                )

                continue

            # Keep this deterministic.
            # The expensive reasoning is performed once
            # during cross-paper synthesis.
            parts = [
                f"Paper {paper_id}: {paper_name}",
                "",
            ]

            categories = (
                paper_data.get(
                    "categories",
                    {},
                )
            )

            for category in self.CATEGORIES:

                category_chunks = (
                    categories.get(
                        category,
                        [],
                    )
                )

                if not category_chunks:

                    continue

                parts.append(
                    f"{category}:"
                )

                added = 0

                for chunk in category_chunks:

                    text = (
                        self._extract_text(
                            chunk
                        )
                    )

                    if not text:

                        continue

                    parts.append(
                        "- "
                        + text[
                            :900
                        ]
                    )

                    added += 1

                    if added >= 2:

                        break

                parts.append("")

            analyses[
                paper_id
            ] = "\n".join(
                parts
            )

        return analyses

    # ============================================================
    # BUILD CROSS-PAPER CONTEXT
    # ============================================================

    def _build_cross_paper_context(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        sections = []

        remaining = (
            self.MAX_TOTAL_CONTEXT_CHARS
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
                + "=" * 70
                + "\n"
                + f"PAPER {index}\n"
                + f"ID: {paper_id}\n"
                + f"TITLE: {paper_name}\n"
                + "=" * 70
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

                sections.append(
                    "No relevant evidence available.\n"
                )

                continue

            paper_chars = 0

            for evidence_index, chunk in enumerate(
                chunks,
                start=1,
            ):

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:

                    continue

                available = min(
                    self.MAX_CHARS_PER_EVIDENCE,
                    self.MAX_CHARS_PER_PAPER
                    - paper_chars,
                    remaining
                    - 150,
                )

                if available <= 200:

                    break

                text = text[
                    :available
                ]

                category = (
                    self._classify_evidence(
                        text
                    )
                )

                block = (
                    f"\nEvidence {evidence_index}"
                    f" [{category}]:\n"
                    f"{text}\n"
                )

                sections.append(
                    block
                )

                length = len(
                    block
                )

                paper_chars += length
                remaining -= length

                if (
                    paper_chars
                    >= self.MAX_CHARS_PER_PAPER
                ):

                    break

                if remaining <= 300:

                    break

            if remaining <= 300:

                break

        return "".join(
            sections
        )[
            :self.MAX_TOTAL_CONTEXT_CHARS
        ]

    # ============================================================
    # CROSS-PAPER SYNTHESIS
    # ============================================================

    def _generate_cross_paper_synthesis(
        self,
        paper_ids: List[int],
        paper_analyses: Dict[int, str],
        evidence: Dict[int, Dict[str, Any]],
    ) -> Optional[str]:

        context = (
            self._build_cross_paper_context(
                paper_ids,
                evidence,
            )
        )

        analysis_context = (
            self._format_paper_analyses(
                paper_ids,
                paper_analyses,
            )
        )

        prompt = (
            self._build_cross_paper_prompt(
                paper_ids=paper_ids,
                context=context,
                analyses=analysis_context,
            )
        )

        return (
            self._call_llm(
                prompt,
                max_tokens=(
                    self.FINAL_SYNTHESIS_MAX_TOKENS
                ),
            )
        )

    # ============================================================
    # CROSS-PAPER PROMPT
    # ============================================================

    def _build_cross_paper_prompt(
        self,
        paper_ids: List[int],
        context: str,
        analyses: str,
    ) -> str:

        paper_names = []

        for paper_id in paper_ids:

            paper_names.append(
                f"Paper {paper_id}"
            )

        return f"""
You are an expert academic researcher,
research-methodology specialist, and
literature-review analyst.

You are working on a research-gap analysis
for a graduate-level research project.

Selected papers:

{", ".join(paper_names)}

============================================================
AVAILABLE PAPER EVIDENCE
============================================================

{context}

============================================================
PAPER-LEVEL ORGANIZATION
============================================================

{analyses}

============================================================
CORE TASK
============================================================

Perform a genuine CROSS-PAPER research-gap analysis.

Do NOT simply summarize each paper.

Instead determine:

1. What has already been solved?
2. What approaches are already well established?
3. Which limitations remain?
4. Which limitations are shared across papers?
5. Which limitations are solved by another selected paper?
6. Which problems remain insufficiently addressed?
7. What meaningful research opportunities remain?

The strongest gap should emerge from comparison
between the selected studies.

============================================================
VERY IMPORTANT
============================================================

A missing detail from the available evidence does
NOT prove that the original paper does not contain it.

Therefore never write:

"The literature has completely ignored X"

unless the evidence genuinely supports such a statement.

Prefer precise language such as:

"The selected studies provide limited evidence regarding X."

or:

"Across the selected studies, X remains insufficiently evaluated."

============================================================
DISTINGUISH THESE FOUR THINGS
============================================================

### 1. Reported Limitation

A limitation explicitly stated by a paper.

### 2. Reported Future Work

A future direction explicitly stated by a paper.

### 3. Evidence-Supported Inference

A reasonable conclusion derived directly from
the supplied evidence.

### 4. Proposed Research Direction

A recommendation for a future study.

Never present category 3 or 4 as though it were
explicitly reported by the authors.

============================================================
WHAT IS NOT AUTOMATICALLY A RESEARCH GAP
============================================================

Do NOT call something a research gap merely because:

- two papers use different models;
- two papers use different datasets;
- one paper has higher accuracy;
- one paper uses CNN while another uses Transformer;
- a paper does not mention something in one passage;
- one study has a limitation that another selected study solves.

The proposed gap must survive cross-paper comparison.

Ask:

"Does another selected paper already address this issue?"

If yes, refine the gap.

============================================================
STRONG GAP INDICATORS
============================================================

Prioritize gaps supported by:

- explicitly reported limitations;
- explicitly reported future work;
- repeated limitations;
- incomplete validation;
- poor generalization;
- limited external validation;
- limited dataset diversity;
- limited population diversity;
- insufficient robustness;
- weak cross-dataset evaluation;
- unresolved methodological problems;
- inconsistent findings;
- limited practical or clinical validation;
- partially solved research problems.

============================================================
REQUIRED OUTPUT
============================================================

# Research Gap Analysis

## 1. Research Landscape

Write 2–3 analytical paragraphs explaining
the research landscape represented by the selected papers.

Explain:

- the common research problem;
- what researchers have already achieved;
- major methodological trends;
- important differences;
- the overall maturity of the research area.

Do not write a paper-by-paper list.

## 2. Cross-Paper Synthesis

Write several analytical paragraphs comparing
the papers directly.

Discuss:

- objectives;
- methodology;
- datasets;
- populations;
- models;
- evaluation;
- findings;
- limitations;
- generalization;
- robustness;
- practical applicability.

Explicitly explain where one study complements,
extends, or differs from another.

## 3. What the Literature Already Solves

Clearly explain which parts of the research problem
are already reasonably addressed.

This section is important because a research gap
must be based on what remains unresolved after
considering existing work.

## 4. Remaining Limitations

Identify limitations that remain after comparing
the selected papers.

Separate:

### Reported Limitations

Only limitations explicitly supported by the papers.

### Evidence-Supported Limitations

Reasonable interpretations derived directly from
the available evidence.

Do not exaggerate.

## 5. Key Research Gaps

Identify the strongest 2–5 genuine research gaps.

Do not force five gaps.

For every gap use:

### Gap 1 — [Short Descriptive Title]

**Evidence**

Identify the paper or papers supporting the gap.

**Current State**

Explain what existing studies have already achieved.

**Unresolved Problem**

Explain precisely what remains insufficiently addressed.

**Why It Matters**

Explain scientific or practical importance.

**Research Opportunity**

Explain what a future study could investigate.

**Evidence Type**

Choose exactly one:

- Reported Limitation
- Reported Future Work
- Evidence-Supported Inference

Repeat for each defensible gap.

## 6. Strongest Research Gap

Select ONE strongest gap.

Explain:

- why it is the strongest;
- which papers support it;
- what existing studies already solve;
- why they do not completely solve this issue;
- why a new study could make a meaningful contribution.

## 7. Recommended Research Direction

Recommend a realistic research direction.

Include:

### Research Objective

What the future study should investigate.

### Possible Methodology

What methodological approach could address it.

### Data Requirements

What type of data would be required.

Do NOT invent a specific dataset unless supported
by the paper evidence.

### Evaluation Strategy

Explain how the proposed approach should be evaluated.

### Expected Contribution

Explain the expected methodological,
scientific, and practical contribution.

## 8. Potential Research Contribution

Explain what a successful future study could contribute.

Discuss:

- methodological contribution;
- scientific contribution;
- practical contribution.

Do not claim guaranteed novelty.

## 9. Research Questions

Generate 2–4 strong research questions that
logically follow from the strongest research gap.

They should be suitable for:

- thesis research;
- research paper;
- graduate research proposal.

## 10. Final Research Gap Statement

Write ONE polished academic paragraph that
could be adapted for the Research Gap section
of a research paper.

It should:

- summarize the current literature;
- identify the unresolved problem;
- explain why it matters;
- establish the need for further research.

Avoid exaggerated claims.

## 11. Confidence Assessment

Choose:

**High**

**Medium**

or

**Low**

Then briefly explain the confidence based on:

- number of selected papers;
- consistency of evidence;
- explicit limitations;
- explicit future work;
- strength of cross-paper agreement.

============================================================
ACADEMIC STYLE
============================================================

Use professional graduate-level academic English.

Write analytical connected paragraphs.

Avoid repetitive paper summaries.

Use transitions naturally:

"Across the selected studies..."

"Collectively..."

"However..."

"In contrast..."

"Taken together..."

"An important distinction is..."

"The comparison indicates..."

"These findings suggest..."

============================================================
EVIDENCE DISCIPLINE
============================================================

Do not invent:

- authors;
- publication years;
- datasets;
- sample sizes;
- models;
- algorithms;
- metrics;
- numerical results;
- experiments;
- limitations;
- future work.

If something cannot be established, say:

"Not established from the available paper evidence."

General academic knowledge may be used to explain
why an established methodological issue matters,
but it must never be used to fabricate paper-specific facts.

============================================================
DO NOT MENTION INTERNAL PROCESSING
============================================================

Do not mention:

- LLM;
- language model;
- prompt;
- embeddings;
- vector database;
- Qdrant;
- retrieval;
- chunks;
- internal processing;
- system instructions;
- ResearchGPT implementation.

Return ONLY the academic research-gap analysis.
"""

    # ============================================================
    # FORMAT PAPER ANALYSES
    # ============================================================

    def _format_paper_analyses(
        self,
        paper_ids: List[int],
        paper_analyses: Dict[int, str],
    ) -> str:

        sections = []

        for paper_id in paper_ids:

            analysis = (
                paper_analyses.get(
                    paper_id,
                    "",
                )
            )

            if not analysis:

                analysis = (
                    "No paper-level evidence summary "
                    "was generated."
                )

            sections.append(
                "\n"
                + "=" * 60
                + f"\nPAPER {paper_id}\n"
                + "=" * 60
                + "\n"
                + analysis
            )

        return "\n".join(
            sections
        )

    # ============================================================
    # RETRY SYNTHESIS
    # ============================================================

    def _generate_retry_synthesis(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> Optional[str]:

        context = (
            self._build_cross_paper_context(
                paper_ids,
                evidence,
            )
        )

        prompt = f"""
You are an academic research-gap analyst.

Analyze the following selected research papers
and produce a concise but useful cross-paper
research-gap analysis.

PAPERS:
{", ".join(
    f"Paper {paper_id}"
    for paper_id in paper_ids
)}

EVIDENCE:
{context}

Your goal is NOT to summarize the papers.

Identify:

1. What the papers collectively solve.
2. What remains unresolved.
3. Which limitations are explicitly reported.
4. Which future directions are explicitly reported.
5. Which evidence-supported gaps remain after
   comparing the papers.

Then provide:

# Research Gap Analysis

## Research Landscape

## What Existing Studies Solve

## Remaining Limitations

## Key Research Gaps

## Strongest Research Gap

## Recommended Research Direction

## Research Questions

## Final Research Gap Statement

## Confidence Assessment

For every paper-specific claim, rely on the
available evidence.

Do not invent facts.

A missing passage does not prove that a paper
does not address something.

Do not claim that no research exists.

Use professional academic English.

Return only the final academic analysis.
"""

        return (
            self._call_llm(
                prompt,
                max_tokens=(
                    self.RETRY_MAX_TOKENS
                ),
            )
        )

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 3000,
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
        # Preferred call
        # --------------------------------------------------------

        try:

            response = method(
                prompt,
                max_tokens=max_tokens,
            )

            extracted = (
                self._extract_llm_response(
                    response
                )
            )

            if extracted:

                return extracted

        except TypeError:

            # Compatibility with implementations
            # that do not accept max_tokens.
            try:

                response = method(
                    prompt
                )

                extracted = (
                    self._extract_llm_response(
                        response
                    )
                )

                if extracted:

                    return extracted

            except Exception as exc:

                print(
                    "ResearchGPT | "
                    f"LLM compatibility call failed: {exc}"
                )

        except Exception as exc:

            print(
                "ResearchGPT | "
                f"LLM generation failed: {exc}"
            )

        return None

    # ============================================================
    # EXTRACT LLM RESPONSE
    # ============================================================

    def _extract_llm_response(
        self,
        response: Any,
    ) -> Optional[str]:

        if isinstance(
            response,
            str,
        ):

            value = (
                response.strip()
            )

            return value or None

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

        return None

    # ============================================================
    # CLEAN LLM OUTPUT
    # ============================================================

    def _clean_text_output(
        self,
        text: Any,
    ) -> str:

        if text is None:

            return ""

        text = str(
            text
        ).strip()

        # Remove accidental markdown fences.
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

        # Remove accidental response prefixes.
        text = re.sub(
            r"^\s*(answer|response|final answer)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove thinking blocks.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        # Remove excessive blank lines.
        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # FALLBACK REPORT
    # ============================================================

    def _fallback_report(
        self,
        paper_ids: List[int],
        paper_analyses: Dict[int, str],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        sections = [
            "# Research Gap Analysis",
            "",
            (
                "The following analysis is based on the "
                "available evidence from the selected "
                "research papers. A complete cross-paper "
                "synthesis could not be generated, so "
                "unsupported research gaps are not asserted."
            ),
            "",
            "## 1. Research Landscape",
            "",
            (
                f"The selected literature contains "
                f"{len(paper_ids)} paper(s). The available "
                "evidence provides information concerning "
                "their research objectives, methodologies, "
                "data, computational approaches, findings, "
                "limitations, and future directions."
            ),
        ]

        # --------------------------------------------------------
        # What is already covered
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## 2. What Existing Studies Solve",
                "",
            ]
        )

        for paper_id in paper_ids:

            paper_data = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            chunks = (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            )

            if not chunks:

                continue

            sections.append(
                f"### Paper {paper_id}"
            )

            added = 0

            for chunk in chunks[:3]:

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:

                    continue

                sections.append(
                    text[:1200]
                )

                added += 1

                if added >= 3:

                    break

        # --------------------------------------------------------
        # Limitations
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## 3. Remaining Limitations",
                "",
            ]
        )

        limitation_count = 0

        for paper_id in paper_ids:

            categories = (
                evidence.get(
                    paper_id,
                    {},
                ).get(
                    "categories",
                    {},
                )
            )

            limitations = (
                categories.get(
                    "Limitations and Failure Cases",
                    [],
                )
            )

            for chunk in limitations[:3]:

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:

                    continue

                sections.append(
                    f"**Paper {paper_id}:** "
                    f"{text[:1500]}"
                )

                limitation_count += 1

        if limitation_count == 0:

            sections.append(
                (
                    "No explicit limitations were established "
                    "from the available evidence."
                )
            )

        # --------------------------------------------------------
        # Future work
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## 4. Reported Future Work",
                "",
            ]
        )

        future_count = 0

        for paper_id in paper_ids:

            categories = (
                evidence.get(
                    paper_id,
                    {},
                ).get(
                    "categories",
                    {},
                )
            )

            future = (
                categories.get(
                    "Future Work and Recommendations",
                    [],
                )
            )

            for chunk in future[:3]:

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:

                    continue

                sections.append(
                    f"**Paper {paper_id}:** "
                    f"{text[:1500]}"
                )

                future_count += 1

        if future_count == 0:

            sections.append(
                (
                    "No explicit future-work directions "
                    "were established from the available evidence."
                )
            )

        # --------------------------------------------------------
        # Gap caution
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## 5. Key Research Gaps",
                "",
                (
                    "A definitive cross-paper research gap "
                    "could not be established automatically "
                    "from the available evidence. The reported "
                    "limitations and future directions above "
                    "should be compared across the complete "
                    "papers before making a strong novelty claim."
                ),
                "",
                "## 6. Strongest Research Gap",
                "",
                (
                    "Not established from the available "
                    "paper evidence."
                ),
                "",
                "## 7. Recommended Research Direction",
                "",
                (
                    "A future study should investigate the "
                    "limitations and unresolved directions "
                    "identified across the selected papers, "
                    "with particular attention to areas where "
                    "the existing evidence remains incomplete."
                ),
                "",
                "## 8. Research Questions",
                "",
                (
                    "1. How can the limitations identified "
                    "across the selected studies be addressed?"
                ),
                (
                    "2. How can evaluation be strengthened "
                    "to improve confidence in generalization?"
                ),
                (
                    "3. Which methodological improvements "
                    "could address the unresolved problems?"
                ),
                "",
                "## 9. Final Research Gap Statement",
                "",
                (
                    "The selected studies demonstrate progress "
                    "toward addressing their respective research "
                    "problems; however, the available evidence "
                    "also indicates limitations and future "
                    "directions that warrant further investigation. "
                    "A definitive research-gap statement should "
                    "be formulated only after confirming that "
                    "the identified limitation remains unresolved "
                    "across the complete selected literature."
                ),
                "",
                "## 10. Confidence Assessment",
                "",
                "**Low**",
                "",
                (
                    "Confidence is limited because the fallback "
                    "analysis is restricted to the evidence "
                    "available to the system and does not perform "
                    "the full cross-paper academic synthesis."
                ),
            ]
        )

        return "\n".join(
            sections
        )

    # ============================================================
    # FLATTEN SOURCES
    # ============================================================

    def _flatten_sources(
        self,
        evidence: Dict[
            int,
            Dict[str, Any],
        ],
    ) -> List[Dict[str, Any]]:

        sources = []

        seen = set()

        for paper_id, paper_data in (
            evidence.items()
        ):

            paper_name = (
                paper_data.get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
            )

            chunks = (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            )

            for chunk in chunks:

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:

                    continue

                fingerprint = (
                    paper_id,
                    self._normalize_for_comparison(
                        text
                    )[:1000],
                )

                if fingerprint in seen:

                    continue

                seen.add(
                    fingerprint
                )

                source = dict(
                    chunk
                )

                source[
                    "paper_id"
                ] = paper_id

                source[
                    "paper_name"
                ] = paper_name

                source[
                    "category"
                ] = self._classify_evidence(
                    text
                )

                source[
                    "text"
                ] = text

                sources.append(
                    source
                )

        return sources


# ================================================================
# SERVICE INSTANCE
# ================================================================

research_gap_service = (
    ResearchGapService()
)