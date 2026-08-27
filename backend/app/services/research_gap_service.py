from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from app.services.multi_document_service import multi_document_service
from app.services.chat_service import chat_service


class ResearchGapService:
    """
    PaperAxiom Research Gap Service.

    Generates evidence-grounded, cross-paper research-gap analysis.

    Main goals:
        1. Understand every selected paper.
        2. Compare papers rather than summarize them independently.
        3. Identify genuine research gaps.
        4. Distinguish reported limitations from model inference.
        5. Use the LLM for academic synthesis and reasoning.
        6. Preserve the existing backend API.

    Existing interface:
        research_gap_service.generate(
            paper_ids=[...]
        )
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MAX_PAPERS = 10

    # More evidence gives the LLM a much stronger understanding
    # of each paper.
    RETRIEVAL_LIMIT = 10

    EVIDENCE_PER_QUERY = 4

    MAX_EVIDENCE_PER_PAPER = 30

    # Maximum source context sent to paper-level analysis.
    PAPER_SOURCE_MAX_CHARS = 26000

    # Maximum source context used during cross-paper synthesis.
    CROSS_SOURCE_MAX_CHARS = 36000

    # Larger outputs for serious academic analysis.
    PAPER_ANALYSIS_MAX_TOKENS = 3000

    FINAL_SYNTHESIS_MAX_TOKENS = 5000

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

    CATEGORY_QUERIES = {
        "Research Problem and Objective": (
            "research problem research question objective aim "
            "purpose study motivation hypothesis scope "
            "problem statement research challenge"
        ),

        "Background and Motivation": (
            "background motivation literature context "
            "clinical scientific practical motivation "
            "why this problem matters research need"
        ),

        "Methodology and Experimental Design": (
            "method methodology research design experimental setup "
            "workflow pipeline procedure preprocessing training "
            "validation testing implementation experiment"
        ),

        "Data, Dataset and Population": (
            "dataset data source population participants patients "
            "samples sample size images videos records annotations "
            "labels data collection train validation test split"
        ),

        "Models, Algorithms and Techniques": (
            "model architecture algorithm framework machine learning "
            "deep learning neural network CNN Transformer "
            "feature extraction classification segmentation "
            "detection optimization"
        ),

        "Results and Evaluation": (
            "results findings performance evaluation accuracy "
            "precision recall F1 AUC Dice IoU sensitivity "
            "specificity error statistical significance conclusion"
        ),

        "Comparison with Existing Methods": (
            "comparison baseline existing method previous work "
            "state of the art benchmark superior inferior "
            "performance comparison ablation experiment"
        ),

        "Contribution and Novelty": (
            "contribution novelty innovation proposed approach "
            "original contribution advancement significance "
            "improvement new framework"
        ),

        "Limitations and Failure Cases": (
            "limitations weaknesses constraints failure cases "
            "generalization robustness bias small dataset "
            "computational limitations unresolved problems"
        ),

        "Future Work and Recommendations": (
            "future work future research recommendations "
            "extension improvement next steps unresolved "
            "research opportunities"
        ),
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
        Generate a complete cross-paper research-gap analysis.

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
        # Maximum number of papers
        # --------------------------------------------------------

        if len(normalized_ids) > self.MAX_PAPERS:

            normalized_ids = normalized_ids[
                :self.MAX_PAPERS
            ]

        print(
            "=================================================="
        )

        print(
            "Research Gap | "
            f"Starting analysis for "
            f"{len(normalized_ids)} paper(s)"
        )

        # ========================================================
        # STEP 1 — RETRIEVE BROAD EVIDENCE
        # ========================================================

        evidence = (
            self._retrieve_evidence(
                normalized_ids
            )
        )

        sources = (
            self._flatten_sources(
                evidence
            )
        )

        # ========================================================
        # STEP 2 — UNDERSTAND EACH PAPER
        # ========================================================

        paper_analyses: Dict[int, str] = {}

        for paper_id in normalized_ids:

            print(
                "Research Gap | "
                f"Deep analysis of paper {paper_id}"
            )

            paper_evidence = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            paper_analyses[
                paper_id
            ] = self._analyze_paper(
                paper_id=paper_id,
                evidence=paper_evidence,
            )

        # ========================================================
        # STEP 3 — CROSS-PAPER SYNTHESIS
        # ========================================================

        print(
            "Research Gap | "
            "Performing cross-paper synthesis"
        )

        research_gap = (
            self._generate_cross_paper_synthesis(
                paper_ids=normalized_ids,
                paper_analyses=paper_analyses,
                evidence=evidence,
            )
        )

        # ========================================================
        # STEP 4 — FALLBACK
        # ========================================================

        if not research_gap:

            print(
                "Research Gap | "
                "LLM synthesis unavailable; "
                "using evidence-grounded fallback"
            )

            research_gap = (
                self._fallback_report(
                    paper_ids=normalized_ids,
                    paper_analyses=paper_analyses,
                    evidence=evidence,
                )
            )

        print(
            "Research Gap | Analysis completed"
        )

        print(
            "=================================================="
        )

        return {
            "research_gap": research_gap,

            "papers_count": len(
                normalized_ids
            ),

            "source_count": len(
                sources
            ),

            "sources": sources,

            "paper_analyses": (
                paper_analyses
            ),

            "evidence": evidence,
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
    # RETRIEVE EVIDENCE
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:

        evidence = {}

        for paper_id in paper_ids:

            paper_data = {
                "categories": {},
                "all_chunks": [],
                "paper_name": (
                    f"Paper {paper_id}"
                ),
            }

            # ----------------------------------------------------
            # Run every research dimension.
            # ----------------------------------------------------

            for category in self.CATEGORIES:

                query = (
                    self.CATEGORY_QUERIES.get(
                        category,
                        category,
                    )
                )

                print(
                    "Research Gap | Retrieval | "
                    f"Paper {paper_id} | "
                    f"{category}"
                )

                results = (
                    self._retrieve_category(
                        query=query,
                        paper_ids=[
                            paper_id
                        ],
                    )
                )

                selected = (
                    self._select_evidence(
                        results,
                        limit=(
                            self.EVIDENCE_PER_QUERY
                        ),
                    )
                )

                paper_data[
                    "categories"
                ][category] = selected

                paper_data[
                    "all_chunks"
                ].extend(
                    selected
                )

                # ------------------------------------------------
                # Try to recover actual title.
                # ------------------------------------------------

                if selected:

                    for item in selected:

                        detected_name = (
                            self._detect_paper_name(
                                item
                            )
                        )

                        if detected_name:

                            paper_data[
                                "paper_name"
                            ] = detected_name

                            break

            # ----------------------------------------------------
            # Remove duplicates.
            # ----------------------------------------------------

            deduplicated = (
                self._deduplicate_chunks(
                    paper_data[
                        "all_chunks"
                    ]
                )
            )

            # ----------------------------------------------------
            # Re-rank final evidence.
            # ----------------------------------------------------

            deduplicated.sort(
                key=self._result_score,
                reverse=True,
            )

            paper_data[
                "all_chunks"
            ] = deduplicated[
                :self.MAX_EVIDENCE_PER_PAPER
            ]

            evidence[
                paper_id
            ] = paper_data

            print(
                "Research Gap | Paper "
                f"{paper_id} | "
                f"{len(paper_data['all_chunks'])} "
                "evidence items"
            )

        return evidence

    # ============================================================
    # RETRIEVAL ADAPTER
    # ============================================================

    def _retrieve_category(
        self,
        query: str,
        paper_ids: List[int],
    ) -> List[Dict[str, Any]]:

        service = (
            self.multi_document_service
        )

        # Existing implementation may expose different
        # method names. Try them safely.
        methods = [
            "search",
            "search_multiple_papers",
            "retrieve",
            "retrieve_multi_document",
            "multi_document_search",
        ]

        for method_name in methods:

            method = getattr(
                service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            attempts = [
                {
                    "query": query,
                    "paper_ids": paper_ids,
                    "limit": (
                        self.RETRIEVAL_LIMIT
                    ),
                },
                {
                    "query": query,
                    "paper_ids": paper_ids,
                    "limit_per_paper": (
                        self.RETRIEVAL_LIMIT
                    ),
                },
                {
                    "query": query,
                    "paper_ids": paper_ids,
                },
                {
                    "question": query,
                    "paper_ids": paper_ids,
                    "limit": (
                        self.RETRIEVAL_LIMIT
                    ),
                },
            ]

            for kwargs in attempts:

                try:

                    result = method(
                        **kwargs
                    )

                    normalized = (
                        self._normalize_results(
                            result
                        )
                    )

                    if normalized:

                        return normalized

                except TypeError:

                    # Signature mismatch.
                    continue

                except Exception as exc:

                    print(
                        "Research gap retrieval "
                        "warning | "
                        f"{method_name}: "
                        f"{exc}"
                    )

                    break

        return []

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
            dict,
        ):

            for key in (
                "results",
                "sources",
                "chunks",
                "documents",
                "evidence",
                "papers",
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

            # The object itself may be a document.
            if self._extract_text(
                result
            ):

                return [
                    result
                ]

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

        return []

    # ============================================================
    # EVIDENCE SELECTION
    # ============================================================

    def _select_evidence(
        self,
        results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:

        if not results:
            return []

        ranked = sorted(
            results,
            key=self._result_score,
            reverse=True,
        )

        selected = []

        seen = set()

        for item in ranked:

            text = (
                self._extract_text(
                    item
                )
            )

            if not text:
                continue

            normalized = (
                self._normalize_for_comparison(
                    text
                )
            )

            fingerprint = (
                normalized[:1000]
            )

            if fingerprint in seen:
                continue

            seen.add(
                fingerprint
            )

            copied = dict(
                item
            )

            copied[
                "text"
            ] = text

            selected.append(
                copied
            )

            if len(
                selected
            ) >= limit:

                break

        return selected

    # ============================================================
    # RESULT SCORE
    # ============================================================

    def _result_score(
        self,
        item: Dict[str, Any],
    ) -> float:

        for key in (
            "score",
            "similarity",
            "relevance_score",
            "rerank_score",
            "distance",
        ):

            value = item.get(
                key
            )

            try:

                score = float(
                    value
                )

                # Lower distance is better.
                if key == "distance":

                    return -score

                return score

            except (
                TypeError,
                ValueError,
            ):

                continue

        return 0.0

    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    def _normalize_for_comparison(
        self,
        text: str,
    ) -> str:

        text = (
            text
            .replace(
                "\x00",
                " ",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        return " ".join(
            text.lower().split()
        )

    # ============================================================
    # DEDUPLICATE
    # ============================================================

    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        result = []

        seen = []

        for chunk in chunks:

            text = (
                self._extract_text(
                    chunk
                )
            )

            if not text:
                continue

            normalized = (
                self._normalize_for_comparison(
                    text
                )
            )

            if not normalized:
                continue

            # Exact / near duplicate.
            duplicate = False

            for old in seen:

                if (
                    normalized
                    == old
                ):

                    duplicate = True
                    break

                # Compare first 700 characters.
                if (
                    len(normalized) > 700
                    and len(old) > 700
                    and normalized[:700]
                    == old[:700]
                ):

                    duplicate = True
                    break

            if duplicate:
                continue

            seen.append(
                normalized
            )

            copied = dict(
                chunk
            )

            copied[
                "text"
            ] = text

            result.append(
                copied
            )

        return result

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================

    def _extract_text(
        self,
        item: Dict[str, Any],
    ) -> str:

        for key in (
            "text",
            "content",
            "chunk",
            "page_content",
            "document",
            "body",
            "snippet",
        ):

            value = item.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                text = (
                    value
                    .replace(
                        "\x00",
                        " ",
                    )
                    .strip()
                )

                if text:
                    return text

        return ""

    # ============================================================
    # PAPER NAME
    # ============================================================

    def _detect_paper_name(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:

        possible_keys = (
            "paper_name",
            "paper_title",
            "title",
            "document_title",
            "filename",
            "file_name",
            "name",
            "source",
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

                    value = value.strip()

                    if value:
                        return value

        return None

    # ============================================================
    # PAPER-LEVEL ANALYSIS
    # ============================================================

    def _analyze_paper(
        self,
        paper_id: int,
        evidence: Dict[str, Any],
    ) -> str:

        if not evidence:

            return (
                f"Paper {paper_id}: "
                "No evidence was retrieved."
            )

        paper_name = (
            evidence.get(
                "paper_name",
                f"Paper {paper_id}",
            )
        )

        evidence_text = (
            self._format_paper_evidence(
                paper_id=paper_id,
                evidence=evidence,
            )
        )

        prompt = f"""
You are an expert academic researcher working inside PaperAxiom.

Your task is to understand ONE research paper deeply before
comparing it with other papers.

============================================================
PAPER
============================================================

{paper_name}

Paper ID: {paper_id}

============================================================
RETRIEVED PAPER EVIDENCE
============================================================

{evidence_text}

============================================================
IMPORTANT
============================================================

The supplied evidence comes from the selected research paper.

Use it as the primary factual source.

Your job is NOT to copy the evidence.

Instead, synthesize it into a coherent academic understanding.

You may use your own academic knowledge to explain why a method,
limitation, evaluation choice or contribution matters.

However, never invent paper-specific facts.

Do not invent:

- datasets
- sample sizes
- models
- algorithms
- numerical results
- experiments
- authors
- publication details
- limitations
- future work

Missing evidence does NOT prove that something is absent from
the original paper.

If something cannot be established, explicitly state:

"Not established from the available evidence."

============================================================
CRITICAL DISTINCTIONS
============================================================

Keep these separate:

1. Reported limitation
2. Reported future work
3. Evidence-supported interpretation
4. Your proposed research opportunity

A methodological difference is NOT automatically a research gap.

A missing retrieved passage is NOT automatically a limitation.

============================================================
ANALYSIS
============================================================

Write a detailed academic analysis using the following structure.

# Paper Understanding

## 1. Research Problem and Objective

Explain:

- what problem the paper addresses
- why the problem matters
- research objective/question
- motivation
- scope

## 2. Methodology

Explain:

- research design
- workflow
- preprocessing
- algorithms
- architecture
- training/testing
- experimental setup

## 3. Data and Dataset

Explain:

- data source
- dataset
- population
- sample characteristics
- labels
- train/validation/test strategy

Only include details supported by evidence.

## 4. Models and Techniques

Explain the major models, algorithms,
architectures and techniques.

Also explain why they are used when this can be reasonably
interpreted from the evidence.

## 5. Results and Evaluation

Explain:

- major findings
- evaluation metrics
- comparisons
- statistical results
- important numerical performance

Preserve numbers accurately.

## 6. Contribution and Novelty

Explain:

- what the study adds
- what is distinctive
- what improvement it provides
- why the contribution matters

## 7. Limitations

Separate:

### Reported Limitations

Only limitations supported by the paper.

### Evidence-Supported Limitations

Reasonable limitations that follow directly from the supplied
evidence.

Clearly label them as interpretation.

## 8. Future Work

Separate:

### Reported Future Work

Directions explicitly stated by the paper.

### Additional Research Opportunities

Potential opportunities that can reasonably be inferred from
the evidence.

## 9. Unresolved Issues

Identify the most important questions that remain unresolved.

These unresolved issues will later be compared against the
other selected papers.

## 10. Research-Gap-Relevant Summary

Finish with a concise paragraph explaining:

- what the paper solves
- what it does not fully solve
- what future research should investigate

============================================================
STYLE
============================================================

Use graduate-level academic English.

Write connected paragraphs.

Be analytical rather than descriptive.

Do not mention:

- LLM
- prompt
- embeddings
- vector database
- retrieval
- chunks
- internal processing
- PaperAxiom system

The final text should read like a genuine academic literature
analysis.
"""

        answer = self._call_llm(
            prompt,
            max_tokens=(
                self.PAPER_ANALYSIS_MAX_TOKENS
            ),
        )

        if answer:

            return self._clean_text_output(
                answer
            )

        return (
            self._deterministic_paper_analysis(
                paper_id=paper_id,
                evidence=evidence,
            )
        )

    # ============================================================
    # CROSS-PAPER SYNTHESIS
    # ============================================================

    def _generate_cross_paper_synthesis(
        self,
        paper_ids: List[int],
        paper_analyses: Dict[int, str],
        evidence: Dict[int, Dict[str, Any]],
    ) -> Optional[str]:

        analysis_text = (
            self._format_paper_analyses(
                paper_ids,
                paper_analyses,
            )
        )

        evidence_text = (
            self._format_cross_paper_evidence(
                paper_ids,
                evidence,
            )
        )

        prompt = (
            self._build_cross_paper_prompt(
                paper_ids=paper_ids,
                paper_analyses=analysis_text,
                evidence=evidence_text,
            )
        )

        answer = self._call_llm(
            prompt,
            max_tokens=(
                self.FINAL_SYNTHESIS_MAX_TOKENS
            ),
        )

        if answer:

            return self._clean_text_output(
                answer
            )

        return None

    # ============================================================
    # CROSS-PAPER PROMPT
    # ============================================================

    def _build_cross_paper_prompt(
        self,
        paper_ids: List[int],
        paper_analyses: str,
        evidence: str,
    ) -> str:

        paper_list = ", ".join(
            f"Paper {paper_id}"
            for paper_id in paper_ids
        )

        return f"""
You are an expert academic researcher and literature-review
specialist.

You are analyzing the following selected research papers:

{paper_list}

Your objective is to identify the strongest and most defensible
research gap across these papers.

============================================================
PAPER-LEVEL ANALYSES
============================================================

{paper_analyses}

============================================================
SUPPORTING EVIDENCE
============================================================

{evidence}

============================================================
YOUR CORE TASK
============================================================

Do NOT summarize the papers independently.

Perform a genuine CROSS-PAPER ANALYSIS.

You must reason about how the studies relate to one another.

Specifically compare:

- research objectives
- research problems
- datasets
- populations
- methodology
- model architectures
- algorithms
- experimental design
- evaluation strategies
- reported performance
- baseline comparisons
- generalization
- robustness
- limitations
- future work
- unresolved issues
- practical applicability

Then determine what remains insufficiently addressed.

============================================================
WHAT COUNTS AS A STRONG RESEARCH GAP?
============================================================

A strong gap should preferably be supported by one or more of:

1. An explicitly reported limitation.

2. An explicitly reported future-work direction.

3. A repeated limitation across multiple papers.

4. An unresolved problem appearing across studies.

5. A methodological weakness supported by evidence.

6. Weak or incomplete validation.

7. Poor generalization.

8. Dataset limitations.

9. Lack of robustness.

10. Incomplete comparison with relevant approaches.

11. Inconsistent or conflicting findings.

12. A meaningful problem that the existing approaches
    address only partially.

============================================================
VERY IMPORTANT
============================================================

Do NOT automatically call something a gap simply because:

- one paper uses CNN and another uses Transformer;
- one paper uses dataset A and another uses dataset B;
- one paper has better accuracy;
- a topic is not mentioned in one retrieved passage;
- a paper has a limitation that another paper already solves.

The gap must survive CROSS-PAPER comparison.

Before calling something a research gap, mentally ask:

"Does another selected paper already address this problem?"

If yes, refine the gap.

If the papers collectively address it adequately,
do not falsely present it as a gap.

============================================================
EVIDENCE VS INFERENCE
============================================================

Clearly distinguish:

**Reported Limitation**

A limitation explicitly stated by a paper.

**Reported Future Work**

A future direction explicitly stated by a paper.

**Evidence-Supported Inference**

A limitation or opportunity reasonably derived from
the supplied evidence.

**Proposed Research Direction**

Your recommendation for a future study.

Never present an inference as if the original authors reported it.

============================================================
OUTPUT
============================================================

Produce the following complete academic analysis.

# Research Gap Analysis

## 1. Research Landscape

Write 1–2 strong paragraphs explaining the overall research
landscape represented by the selected papers.

Explain:

- what they collectively investigate
- common objectives
- major methodological trends
- major differences

Do NOT simply list the papers.

============================================================

## 2. Cross-Paper Synthesis

Write 3–5 analytical paragraphs.

Compare the studies directly.

Use meaningful transitions such as:

"Across the selected studies..."

"Collectively, the evidence suggests..."

"However, the studies differ in..."

"While Paper X..., Paper Y..."

"Taken together..."

"The comparison reveals..."

Focus on relationships between studies.

============================================================

## 3. Evidence Matrix in Prose

Explain the following comparisons in connected prose:

### Problem Coverage

What aspects of the problem have already been addressed?

### Methodological Coverage

Which methodological approaches are represented?

### Dataset Coverage

What data populations and datasets are covered?

### Evaluation Coverage

How thoroughly are the methods evaluated?

### Generalization and Robustness

What evidence exists for robustness and generalization?

### Practical Applicability

How close are the approaches to practical deployment?

============================================================

## 4. Key Research Gaps

Identify the 2–5 strongest genuine gaps.

Do NOT force five gaps if only two or three are defensible.

For every gap use:

### Gap 1 — [Short Descriptive Title]

**Evidence**

Which paper(s) support this gap?

**Current State**

What has already been solved?

**Unresolved Problem**

What remains insufficiently addressed?

**Why It Matters**

Why is this scientifically or practically important?

**Research Opportunity**

What could a new study investigate?

**Evidence Type**

Choose one:

- Reported Limitation
- Reported Future Work
- Evidence-Supported Inference

Repeat for each genuine gap.

============================================================

## 5. Strongest Research Gap

Select ONE strongest gap.

Explain:

- why it is the most defensible
- which papers support it
- why existing papers do not fully solve it
- what makes it suitable for a new study

============================================================

## 6. Recommended Research Direction

Propose a realistic research direction directly addressing
the strongest gap.

Include:

- research objective
- possible methodology
- data requirements
- evaluation strategy
- expected contribution

Do not invent a specific dataset unless supported by the evidence.

============================================================

## 7. Potential Research Contribution

Explain what a successful future study could contribute.

Cover:

- methodological contribution
- scientific contribution
- practical contribution

============================================================

## 8. Research Questions

Generate 2–4 research questions that logically follow from
the strongest gap.

They should be suitable for a thesis or research paper.

============================================================

## 9. Final Research Gap Statement

Write ONE polished academic paragraph that could be adapted
directly into the "Research Gap" section of a research paper.

It should:

- summarize the current literature
- identify what remains unresolved
- explain why it matters
- establish the need for further research

Do not exaggerate novelty.

============================================================

## 10. Confidence

Provide:

**High**

**Medium**

or

**Low**

Then explain confidence briefly based on:

- number of papers
- consistency of evidence
- explicit limitations
- explicit future work
- strength of cross-paper agreement

============================================================
FINAL QUALITY RULES
============================================================

The answer must be:

- academically professional
- analytical
- evidence-grounded
- cross-paper
- useful for a researcher
- sufficiently detailed
- free from fabricated claims

Do NOT merely summarize.

Do NOT repeat the same information.

Do NOT create a gap just to fill the section.

Do NOT claim "no research exists" unless the evidence genuinely
supports that conclusion.

Do NOT use external facts as paper-specific evidence.

Do NOT mention internal system details.
"""

    # ============================================================
    # FORMAT PAPER EVIDENCE
    # ============================================================

    def _format_paper_evidence(
        self,
        paper_id: int,
        evidence: Dict[str, Any],
    ) -> str:

        sections = []

        paper_name = (
            evidence.get(
                "paper_name",
                f"Paper {paper_id}",
            )
        )

        sections.append(
            f"Paper: {paper_name}"
        )

        categories = (
            evidence.get(
                "categories",
                {},
            )
        )

        total_chars = 0

        for category in self.CATEGORIES:

            chunks = (
                categories.get(
                    category,
                    [],
                )
            )

            sections.append(
                f"\n### {category}"
            )

            if not chunks:

                sections.append(
                    "No relevant evidence was retrieved."
                )

                continue

            for index, chunk in enumerate(
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

                remaining = (
                    self.PAPER_SOURCE_MAX_CHARS
                    - total_chars
                )

                if remaining <= 500:

                    break

                text = text[
                    :min(
                        2600,
                        remaining,
                    )
                ]

                sections.append(
                    f"\nEvidence {index}:\n{text}"
                )

                total_chars += len(
                    text
                )

            if total_chars >= (
                self.PAPER_SOURCE_MAX_CHARS
            ):

                break

        return "\n".join(
            sections
        )

    # ============================================================
    # FORMAT CROSS-PAPER EVIDENCE
    # ============================================================

    def _format_cross_paper_evidence(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        sections = []

        total_chars = 0

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

            sections.append(
                "\n"
                + "=" * 70
                + f"\n{paper_name} | Paper {paper_id}\n"
                + "=" * 70
            )

            chunks = (
                paper_data.get(
                    "all_chunks",
                    [],
                )
            )

            # Strongest evidence first.
            ranked = sorted(
                chunks,
                key=self._result_score,
                reverse=True,
            )

            for index, chunk in enumerate(
                ranked[:14],
                start=1,
            ):

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:
                    continue

                remaining = (
                    self.CROSS_SOURCE_MAX_CHARS
                    - total_chars
                )

                if remaining <= 600:
                    break

                text = text[
                    :min(
                        1800,
                        remaining,
                    )
                ]

                category = (
                    chunk.get(
                        "category",
                        "Research Evidence",
                    )
                )

                sections.append(
                    f"\n[{paper_name}] "
                    f"{category} — Evidence {index}\n"
                    f"{text}"
                )

                total_chars += len(
                    text
                )

            if total_chars >= (
                self.CROSS_SOURCE_MAX_CHARS
            ):

                break

        return "\n".join(
            sections
        )

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
                    "No independent paper analysis "
                    "was generated."
                )

            sections.append(
                "\n"
                + "=" * 70
                + f"\nPAPER {paper_id}\n"
                + "=" * 70
                + "\n"
                + analysis
            )

        return "\n".join(
            sections
        )

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 2000,
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
                "Research Gap | "
                "ChatService._call_llm unavailable."
            )

            return None

        # --------------------------------------------------------
        # First attempt.
        # --------------------------------------------------------

        attempts = [
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
            },
            {
                "prompt": prompt,
            },
        ]

        for kwargs in attempts:

            try:

                response = method(
                    **kwargs
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

                continue

            except Exception as exc:

                print(
                    "Research Gap | "
                    f"LLM generation failed: "
                    f"{exc}"
                )

                break

        return None

    # ============================================================
    # CLEAN LLM OUTPUT
    # ============================================================

    def _clean_text_output(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = str(
            text
        ).strip()

        # Remove markdown code fences accidentally returned.
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

        # Remove accidental Answer/Response prefix.
        text = re.sub(
            r"^\s*(answer|response)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove obvious system leakage.
        text = re.sub(
            r"(?i)\bAs an AI language model\b",
            "",
            text,
        )

        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # DETERMINISTIC PAPER ANALYSIS
    # ============================================================

    def _deterministic_paper_analysis(
        self,
        paper_id: int,
        evidence: Dict[str, Any],
    ) -> str:

        paper_name = (
            evidence.get(
                "paper_name",
                f"Paper {paper_id}",
            )
        )

        categories = (
            evidence.get(
                "categories",
                {},
            )
        )

        sections = [
            "# Paper Understanding",
            "",
            f"**Paper:** {paper_name}",
        ]

        title_map = {
            "Research Problem and Objective":
                "Research Problem and Objective",

            "Background and Motivation":
                "Background and Motivation",

            "Methodology and Experimental Design":
                "Methodology",

            "Data, Dataset and Population":
                "Data and Dataset",

            "Models, Algorithms and Techniques":
                "Models and Techniques",

            "Results and Evaluation":
                "Results and Evaluation",

            "Comparison with Existing Methods":
                "Comparison with Existing Methods",

            "Contribution and Novelty":
                "Contribution and Novelty",

            "Limitations and Failure Cases":
                "Limitations and Failure Cases",

            "Future Work and Recommendations":
                "Future Work and Recommendations",
        }

        for category in self.CATEGORIES:

            sections.append(
                "\n## "
                + title_map.get(
                    category,
                    category,
                )
            )

            chunks = (
                categories.get(
                    category,
                    [],
                )
            )

            if not chunks:

                sections.append(
                    "Not established from the "
                    "available evidence."
                )

                continue

            # Combine up to three strong pieces.
            added = 0

            for chunk in chunks:

                text = (
                    self._extract_text(
                        chunk
                    )
                )

                if not text:
                    continue

                sections.append(
                    text[:1800]
                )

                added += 1

                if added >= 3:
                    break

        return "\n\n".join(
            sections
        )

    # ============================================================
    # FINAL FALLBACK
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
                "available evidence from the selected papers. "
                "A complete cross-paper synthesis could not "
                "be generated, so unsupported research gaps "
                "are not asserted."
            ),
        ]

        # --------------------------------------------------------
        # Extract reported limitations and future work.
        # --------------------------------------------------------

        limitations = []

        future_work = []

        for paper_id in paper_ids:

            paper_data = (
                evidence.get(
                    paper_id,
                    {},
                )
            )

            categories = (
                paper_data.get(
                    "categories",
                    {},
                )
            )

            for category in (
                "Limitations and Failure Cases",
                "Future Work and Recommendations",
            ):

                chunks = (
                    categories.get(
                        category,
                        [],
                    )
                )

                for chunk in chunks[:3]:

                    text = (
                        self._extract_text(
                            chunk
                        )
                    )

                    if not text:
                        continue

                    if category.startswith(
                        "Limitations"
                    ):

                        limitations.append(
                            (
                                paper_id,
                                text[:1200],
                            )
                        )

                    else:

                        future_work.append(
                            (
                                paper_id,
                                text[:1200],
                            )
                        )

        # --------------------------------------------------------
        # Research landscape.
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## Research Landscape",
                "",
                (
                    f"The selected literature contains "
                    f"{len(paper_ids)} research paper(s). "
                    "The paper-level analyses below summarize "
                    "the evidence available for identifying "
                    "unresolved research problems."
                ),
            ]
        )

        # --------------------------------------------------------
        # Evidence-supported limitations.
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## Evidence-Supported Limitations",
                "",
            ]
        )

        if limitations:

            for index, (
                paper_id,
                text,
            ) in enumerate(
                limitations[:8],
                start=1,
            ):

                sections.append(
                    f"### Limitation {index} "
                    f"(Paper {paper_id})"
                )

                sections.append(
                    text
                )

        else:

            sections.append(
                "No explicit limitations were "
                "established from the available evidence."
            )

        # --------------------------------------------------------
        # Future work.
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## Reported Future Work",
                "",
            ]
        )

        if future_work:

            for index, (
                paper_id,
                text,
            ) in enumerate(
                future_work[:8],
                start=1,
            ):

                sections.append(
                    f"### Future Direction {index} "
                    f"(Paper {paper_id})"
                )

                sections.append(
                    text
                )

        else:

            sections.append(
                "No explicit future-work directions "
                "were established from the available evidence."
            )

        # --------------------------------------------------------
        # Paper analyses.
        # --------------------------------------------------------

        sections.extend(
            [
                "",
                "## Paper-Level Analysis",
                "",
            ]
        )

        for paper_id in paper_ids:

            analysis = (
                paper_analyses.get(
                    paper_id,
                    "",
                )
            )

            sections.extend(
                [
                    f"### Paper {paper_id}",
                    "",
                    (
                        analysis
                        or
                        "No analysis was generated."
                    ),
                    "",
                ]
            )

        return "\n".join(
            sections
        )

    # ============================================================
    # SOURCE FLATTENING
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

            categories = (
                paper_data.get(
                    "categories",
                    {},
                )
            )

            paper_name = (
                paper_data.get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
            )

            for category, chunks in (
                categories.items()
            ):

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
                    ] = category

                    source[
                        "text"
                    ] = text

                    sources.append(
                        source
                    )

        return sources


# ================================================================
# SINGLETON
# ================================================================

research_gap_service = (
    ResearchGapService()
)