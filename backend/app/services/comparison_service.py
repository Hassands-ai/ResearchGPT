from __future__ import annotations

from typing import Dict, List, Any, Optional
import re


from app.services.multi_document_service import (
    multi_document_service,
)

from app.services.chat_service import (
    chat_service,
)


class ComparisonService:
    """
    PaperAxiom Academic Research Paper Comparison Service

    Supports:
        2–10 research papers

    Main responsibilities:
        1. Retrieve balanced evidence from every selected paper.
        2. Preserve paper identity and names where available.
        3. Understand the researcher's comparison question.
        4. Perform cross-paper academic synthesis using the LLM.
        5. Produce concise, paragraph-based answers.
        6. Avoid unsupported claims and fabricated paper details.
        7. Provide evidence/source information to the frontend.

    Existing API compatibility:
        comparison_service.compare(
            paper_ids=[...],
            evidence_per_paper=...,
            user_question="..."
        )
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MIN_PAPERS = 2
    MAX_PAPERS = 10

    DEFAULT_EVIDENCE_PER_PAPER = 5

    # Retrieval can return more evidence than before.
    # We later select the strongest unique evidence.
    RETRIEVAL_LIMIT_PER_PAPER = 5

    # Maximum evidence chunks retained per category.
    MAX_CHUNKS_PER_CATEGORY = 3

    # Maximum evidence chunks used for final LLM synthesis.
    MAX_TOTAL_EVIDENCE_PER_PAPER = 18

    # Much larger than the previous 850-token limit.
    # This gives the LLM enough space for an actual comparison.
    LLM_MAX_TOKENS = 2400

    # Maximum prompt context.
    # We avoid the previous aggressive 30k-character cut.
    MAX_CONTEXT_CHARS = 60000

    # ============================================================
    # RESEARCH CATEGORIES
    # ============================================================

    CATEGORIES = [
        (
            "Research Problem & Objective",
            (
                "research problem research question objective aim "
                "purpose motivation hypothesis significance "
                "clinical problem scientific problem"
            ),
        ),

        (
            "Methodology",
            (
                "method methodology research design experimental "
                "procedure workflow framework pipeline protocol "
                "preprocessing training validation testing "
                "implementation"
            ),
        ),

        (
            "Dataset & Data",
            (
                "dataset data source population participants "
                "patients samples images videos annotations labels "
                "data collection preprocessing train validation test "
                "split"
            ),
        ),

        (
            "Models & Algorithms",
            (
                "model architecture algorithm method framework "
                "machine learning deep learning neural network "
                "transformer CNN classification segmentation "
                "detection feature extraction optimization"
            ),
        ),

        (
            "Results & Evaluation",
            (
                "results findings evaluation performance accuracy "
                "precision recall F1 AUC Dice IoU sensitivity "
                "specificity experiments comparison statistical "
                "significance performance"
            ),
        ),

        (
            "Main Contribution & Novelty",
            (
                "contribution novelty innovation proposed approach "
                "improvement advancement significance original "
                "method contribution"
            ),
        ),

        (
            "Limitations",
            (
                "limitations weaknesses constraints failure cases "
                "challenges generalization robustness bias "
                "computational limitation clinical limitation "
                "data limitation"
            ),
        ),

        (
            "Future Work",
            (
                "future work future research recommendations "
                "improvements extensions unresolved problems "
                "next steps"
            ),
        ),
    ]

    # ============================================================
    # QUESTION INTENT KEYWORDS
    # ============================================================

    QUESTION_INTENTS = {
        "methodology": [
            "method",
            "methodology",
            "approach",
            "framework",
            "pipeline",
            "procedure",
            "workflow",
            "architecture",
            "technique",
        ],

        "dataset": [
            "dataset",
            "data",
            "sample",
            "samples",
            "population",
            "patients",
            "participants",
            "images",
            "videos",
        ],

        "model": [
            "model",
            "models",
            "algorithm",
            "algorithms",
            "architecture",
            "network",
            "transformer",
            "cnn",
            "deep learning",
            "machine learning",
        ],

        "results": [
            "result",
            "results",
            "performance",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "dice",
            "iou",
            "finding",
            "findings",
        ],

        "limitation": [
            "limitation",
            "limitations",
            "weakness",
            "weaknesses",
            "problem",
            "problems",
            "drawback",
            "drawbacks",
            "challenge",
            "challenges",
        ],

        "contribution": [
            "contribution",
            "contributions",
            "novel",
            "novelty",
            "innovation",
            "advancement",
            "original",
        ],

        "research_gap": [
            "gap",
            "research gap",
            "future research",
            "unresolved",
            "missing",
            "lack",
            "limitation",
            "opportunity",
        ],

        "general": [
            "compare",
            "comparison",
            "difference",
            "differences",
            "similarity",
            "similarities",
            "better",
            "stronger",
            "best",
        ],
    }

    # ============================================================
    # BASIC TEXT CLEANING
    # ============================================================

    def _clean_text(
        self,
        text: Any,
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\x00", " ")
        )

        # Preserve paragraph information.
        lines = []

        for line in text.split("\n"):

            cleaned = " ".join(
                line.split()
            ).strip()

            if cleaned:
                lines.append(cleaned)

        return "\n".join(lines).strip()

    # ============================================================
    # NORMALIZED SINGLE-LINE TEXT
    # ============================================================

    def _compact_text(
        self,
        text: Any,
    ) -> str:

        return " ".join(
            self._clean_text(text).split()
        ).strip()

    # ============================================================
    # DUPLICATE DETECTION
    # ============================================================

    def _is_duplicate(
        self,
        text: str,
        existing: List[str],
    ) -> bool:

        normalized = self._compact_text(
            text
        ).lower()

        if not normalized:
            return True

        # Fast exact/fingerprint check first.
        fingerprint = normalized[:1000]

        for old in existing:

            old_normalized = self._compact_text(
                old
            ).lower()

            if not old_normalized:
                continue

            if normalized == old_normalized:
                return True

            if (
                fingerprint
                and old_normalized.startswith(
                    fingerprint
                )
            ):
                return True

            # Avoid expensive similarity on very short chunks.
            if (
                len(normalized.split()) < 30
                or len(old_normalized.split()) < 30
            ):
                continue

            words_a = set(
                normalized.split()
            )

            words_b = set(
                old_normalized.split()
            )

            union = words_a | words_b

            if not union:
                continue

            similarity = (
                len(words_a & words_b)
                / len(union)
            )

            if similarity >= 0.90:
                return True

        return False

    # ============================================================
    # SCORE EXTRACTION
    # ============================================================

    def _get_score(
        self,
        item: Dict[str, Any],
    ) -> float:

        for key in (
            "score",
            "similarity",
            "relevance_score",
        ):

            value = item.get(key)

            try:
                return float(value)
            except (
                TypeError,
                ValueError,
            ):
                pass

        # Some vector databases expose distance
        # rather than similarity.
        distance = item.get("distance")

        try:
            return -float(distance)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # ============================================================
    # PAPER NAME EXTRACTION
    # ============================================================

    def _extract_paper_name(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:

        keys = [
            "paper_name",
            "paper_title",
            "title",
            "document_title",
            "filename",
            "file_name",
            "name",
        ]

        for key in keys:

            value = item.get(key)

            if isinstance(value, str):

                value = value.strip()

                if value:
                    return value

        metadata = item.get(
            "metadata"
        )

        if isinstance(metadata, dict):

            for key in keys:

                value = metadata.get(key)

                if isinstance(value, str):

                    value = value.strip()

                    if value:
                        return value

        return None

    # ============================================================
    # PAPER ID EXTRACTION
    # ============================================================

    def _extract_paper_id(
        self,
        item: Dict[str, Any],
    ) -> Optional[int]:

        for key in (
            "paper_id",
            "document_id",
            "source_paper_id",
        ):

            try:
                value = item.get(key)

                if value is not None:
                    return int(value)

            except (
                TypeError,
                ValueError,
            ):
                pass

        metadata = item.get(
            "metadata"
        )

        if isinstance(metadata, dict):

            for key in (
                "paper_id",
                "document_id",
                "source_paper_id",
            ):

                try:

                    value = metadata.get(
                        key
                    )

                    if value is not None:
                        return int(value)

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return None

    # ============================================================
    # EVIDENCE TEXT EXTRACTION
    # ============================================================

    def _extract_evidence_text(
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
        ):

            value = item.get(key)

            if isinstance(value, str):

                value = self._clean_text(
                    value
                )

                if value:
                    return value

        return ""

    # ============================================================
    # COLLECT BALANCED EVIDENCE
    # ============================================================

    def _collect_evidence(
        self,
        paper_ids: List[int],
        evidence_per_paper: int = 5,
        user_question: str = "",
    ) -> Dict[int, Dict[str, Any]]:

        evidence = {}

        for paper_id in paper_ids:

            evidence[paper_id] = {
                "paper_name": f"Paper {paper_id}",
                "categories": {},
                "all_chunks": [],
            }

        # --------------------------------------------------------
        # Detect question intent.
        # --------------------------------------------------------

        intent = self._detect_question_intent(
            user_question
        )

        selected_categories = (
            self._select_categories_for_intent(
                intent
            )
        )

        print(
            "Paper Comparison | "
            f"Question intent: {intent}"
        )

        print(
            "Paper Comparison | "
            "Retrieval categories: "
            f"{', '.join(selected_categories)}"
        )

        # --------------------------------------------------------
        # Category retrieval
        # --------------------------------------------------------

        for category_name, category_query in self.CATEGORIES:

            if (
                category_name
                not in selected_categories
            ):
                continue

            query = category_query

            # Add the user's actual question so retrieval
            # is not purely category based.
            if user_question.strip():

                query = (
                    f"{user_question.strip()} "
                    f"{category_query}"
                )

            print(
                "Paper Comparison | "
                f"Retrieving {category_name}"
            )

            try:

                result = (
                    multi_document_service.search(
                        query=query,
                        paper_ids=paper_ids,
                        limit_per_paper=(
                            self.RETRIEVAL_LIMIT_PER_PAPER
                        ),
                    )
                )

            except Exception as exc:

                print(
                    "Paper Comparison | "
                    f"Retrieval failed for "
                    f"{category_name}: {exc}"
                )

                continue

            results = self._normalize_search_results(
                result
            )

            if not results:
                continue

            # ----------------------------------------------------
            # Separate evidence by paper.
            # ----------------------------------------------------

            for paper_id in paper_ids:

                paper_results = []

                for item in results:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    item_paper_id = (
                        self._extract_paper_id(
                            item
                        )
                    )

                    if item_paper_id is None:
                        continue

                    if (
                        item_paper_id
                        != int(paper_id)
                    ):
                        continue

                    paper_results.append(
                        item
                    )

                # Highest relevance first.
                paper_results.sort(
                    key=self._get_score,
                    reverse=True,
                )

                selected = []

                selected_texts = []

                max_chunks = max(
                    1,
                    min(
                        int(
                            evidence_per_paper
                            or self.DEFAULT_EVIDENCE_PER_PAPER
                        ),
                        self.MAX_CHUNKS_PER_CATEGORY,
                    ),
                )

                for item in paper_results:

                    text = (
                        self._extract_evidence_text(
                            item
                        )
                    )

                    if not text:
                        continue

                    if self._is_duplicate(
                        text,
                        selected_texts,
                    ):
                        continue

                    selected_texts.append(
                        text
                    )

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

                    selected.append(
                        {
                            "text": text[:1800],
                            "score": self._get_score(
                                item
                            ),
                            "paper_id": paper_id,
                            "paper_name": (
                                paper_name
                                or evidence[
                                    paper_id
                                ][
                                    "paper_name"
                                ]
                            ),
                        }
                    )

                    if len(selected) >= max_chunks:
                        break

                if selected:

                    evidence[
                        paper_id
                    ][
                        "categories"
                    ][category_name] = selected

                    evidence[
                        paper_id
                    ][
                        "all_chunks"
                    ].extend(
                        selected
                    )

        # --------------------------------------------------------
        # Final deduplication and balancing.
        # --------------------------------------------------------

        for paper_id in paper_ids:

            paper_data = evidence[
                paper_id
            ]

            unique_chunks = []

            seen_texts = []

            # Prioritize strongest evidence.
            chunks = sorted(
                paper_data[
                    "all_chunks"
                ],
                key=self._get_score,
                reverse=True,
            )

            for chunk in chunks:

                text = chunk.get(
                    "text",
                    "",
                )

                if self._is_duplicate(
                    text,
                    seen_texts,
                ):
                    continue

                seen_texts.append(
                    text
                )

                unique_chunks.append(
                    chunk
                )

                if (
                    len(unique_chunks)
                    >= self.MAX_TOTAL_EVIDENCE_PER_PAPER
                ):
                    break

            paper_data[
                "all_chunks"
            ] = unique_chunks

            print(
                "Paper Comparison | "
                f"{paper_data['paper_name']} | "
                f"{len(unique_chunks)} evidence items"
            )

        return evidence

    # ============================================================
    # NORMALIZE SEARCH RESULTS
    # ============================================================

    def _normalize_search_results(
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

            # A dictionary may itself represent one result.
            if self._extract_evidence_text(
                result
            ):

                return [result]

        return []

    # ============================================================
    # QUESTION INTENT
    # ============================================================

    def _detect_question_intent(
        self,
        question: str,
    ) -> str:

        if not question:
            return "general"

        q = question.lower().strip()

        scores = {
            intent: 0
            for intent in self.QUESTION_INTENTS
        }

        for intent, keywords in (
            self.QUESTION_INTENTS.items()
        ):

            for keyword in keywords:

                if keyword in q:
                    scores[intent] += 1

        # Research-gap questions should win over
        # generic limitation questions.
        if (
            "research gap" in q
            or "future research" in q
            or "unresolved" in q
        ):
            return "research_gap"

        best_intent = max(
            scores,
            key=scores.get,
        )

        if scores[
            best_intent
        ] == 0:

            return "general"

        return best_intent

    # ============================================================
    # CATEGORY SELECTION
    # ============================================================

    def _select_categories_for_intent(
        self,
        intent: str,
    ) -> List[str]:

        all_categories = [
            category_name
            for category_name, _
            in self.CATEGORIES
        ]

        if intent == "methodology":

            return [
                "Research Problem & Objective",
                "Methodology",
                "Models & Algorithms",
                "Results & Evaluation",
                "Limitations",
            ]

        if intent == "dataset":

            return [
                "Research Problem & Objective",
                "Dataset & Data",
                "Methodology",
                "Results & Evaluation",
                "Limitations",
            ]

        if intent == "model":

            return [
                "Methodology",
                "Models & Algorithms",
                "Results & Evaluation",
                "Main Contribution & Novelty",
                "Limitations",
            ]

        if intent == "results":

            return [
                "Models & Algorithms",
                "Dataset & Data",
                "Results & Evaluation",
                "Main Contribution & Novelty",
                "Limitations",
            ]

        if intent == "limitation":

            return [
                "Methodology",
                "Dataset & Data",
                "Results & Evaluation",
                "Limitations",
                "Future Work",
            ]

        if intent == "contribution":

            return [
                "Research Problem & Objective",
                "Methodology",
                "Models & Algorithms",
                "Results & Evaluation",
                "Main Contribution & Novelty",
            ]

        if intent == "research_gap":

            return [
                "Research Problem & Objective",
                "Methodology",
                "Dataset & Data",
                "Models & Algorithms",
                "Results & Evaluation",
                "Limitations",
                "Future Work",
            ]

        return all_categories

    # ============================================================
    # BUILD EVIDENCE CONTEXT
    # ============================================================

    def _build_evidence_context(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        sections = []

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            paper_data = evidence.get(
                paper_id,
                {},
            )

            paper_name = paper_data.get(
                "paper_name",
                f"Paper {paper_id}",
            )

            sections.append(
                "\n"
                + "=" * 70
                + f"\nPAPER {index}\n"
                + f"NAME: {paper_name}\n"
                + f"ID: {paper_id}\n"
                + "=" * 70
            )

            categories = paper_data.get(
                "categories",
                {},
            )

            if not categories:

                sections.append(
                    "No relevant evidence was retrieved."
                )

                continue

            for category_name, _ in self.CATEGORIES:

                chunks = categories.get(
                    category_name,
                    [],
                )

                if not chunks:
                    continue

                sections.append(
                    f"\n[{category_name}]"
                )

                for evidence_index, item in enumerate(
                    chunks,
                    start=1,
                ):

                    text = item.get(
                        "text",
                        "",
                    )

                    if not text:
                        continue

                    sections.append(
                        f"\nEvidence {evidence_index}: "
                        f"{text}"
                    )

        context = "\n".join(
            sections
        )

        # Keep complete sections whenever possible.
        if len(context) <= self.MAX_CONTEXT_CHARS:
            return context

        # If extremely large, retain the beginning and
        # strongest evidence rather than blindly truncating.
        return context[
            :self.MAX_CONTEXT_CHARS
        ]

    # ============================================================
    # PAPER LABELS
    # ============================================================

    def _build_paper_labels(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        lines = []

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            paper_name = (
                evidence.get(
                    paper_id,
                    {},
                ).get(
                    "paper_name",
                    f"Paper {paper_id}",
                )
            )

            lines.append(
                f"Paper {index}: "
                f"{paper_name} "
                f"(ID: {paper_id})"
            )

        return "\n".join(
            lines
        )

    # ============================================================
    # BUILD LLM PROMPT
    # ============================================================

    def _build_prompt(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
        question: str,
        intent: str,
    ) -> str:

        paper_labels = (
            self._build_paper_labels(
                paper_ids,
                evidence,
            )
        )

        evidence_context = (
            self._build_evidence_context(
                paper_ids,
                evidence,
            )
        )

        return f"""
You are PaperAxiom, an expert academic research assistant
specialized in scientific literature analysis.

You are comparing {len(paper_ids)} research papers.

============================================================
SELECTED PAPERS
============================================================

{paper_labels}

============================================================
RESEARCHER QUESTION
============================================================

{question}

============================================================
QUESTION INTENT
============================================================

{intent}

============================================================
PRIMARY EVIDENCE
============================================================

The supplied evidence comes from the selected papers.

Use these papers as the primary factual source.

Your job is NOT to simply repeat retrieved text.

You must understand the papers and synthesize the information
into a useful academic comparison.

============================================================
IMPORTANT REASONING RULES
============================================================

1. Consider EVERY selected paper.

2. Do not compare only Paper 1 and Paper 2 when 3–10 papers
   are selected.

3. Compare papers across the dimensions relevant to the
   researcher's question.

4. Use general academic knowledge only to:
   - explain terminology,
   - clarify concepts,
   - explain implications,
   - connect related findings.

5. Never use general knowledge to invent paper-specific facts.

6. Never invent:
   - authors
   - titles
   - datasets
   - sample sizes
   - models
   - algorithms
   - numerical results
   - experiments
   - contributions
   - limitations
   - future work

7. If a paper does not provide enough evidence for a specific
   comparison, write:

   "Not identified in the available paper evidence."

8. A methodological difference is not automatically a weakness.

9. A higher numerical metric is not automatically evidence that
   one paper is scientifically better.

10. If the researcher asks which paper is better, evaluate the
    papers against the requested criterion.

11. If the evidence does not support a winner, explain why.

12. Distinguish clearly between:
    - reported findings
    - evidence-supported interpretation
    - your proposed research implication

============================================================
HOW TO UNDERSTAND THE QUESTION
============================================================

The researcher may ask the question informally.

Examples:

"compare these papers"

"what is difference between these studies"

"which model is better"

"compare methodology"

"which datasets are used"

"what are the limitations"

"what can I use for my research"

"which paper is more useful for my thesis"

"compare their results"

"what research gap is common"

Interpret the actual intent naturally.

Do not force a generic comparison if the researcher asks a
focused question.

============================================================
CROSS-PAPER SYNTHESIS
============================================================

When comparing papers, explicitly identify:

- common objective
- important differences
- methodological similarities
- methodological differences
- dataset similarities/differences
- model/algorithm differences
- evaluation differences
- result differences
- contribution differences
- limitation differences
- future-work differences

Do not list facts independently.

Explain what those differences mean academically.

For example, instead of:

"Paper 1 uses CNN.
Paper 2 uses Transformer."

Prefer:

"Paper 1 relies on a CNN-based approach, whereas Paper 2
adopts a Transformer-based architecture, indicating a
difference in how the studies model spatial or contextual
information."

Only make the interpretation when supported by the evidence.

============================================================
RESEARCHER-USEFUL SYNTHESIS
============================================================

Where appropriate, explain:

- Which approach appears more suitable for a specific task.
- Which methodology is more comprehensive.
- Which dataset is more representative.
- Which evaluation is more convincing.
- Which limitations remain unresolved.
- What combination of ideas could inform future research.

Do not make unsupported claims.

============================================================
OUTPUT STYLE
============================================================

Write in clear professional academic English.

Keep the answer concise but substantive.

Use paragraph-based explanations.

Avoid unnecessary repetition.

Avoid generic filler.

Do not write a long introduction.

Do not expose:
- chain of thought
- internal reasoning
- prompts
- retrieval process
- embeddings
- vector databases
- model names used internally by PaperAxiom

============================================================
OUTPUT FORMAT
============================================================

For a GENERAL comparison, use:

## Overall Comparison

One concise paragraph explaining the research landscape.

## Research Objectives

Compare the research objectives and problems.

## Methodology

Compare the research methodologies and workflows.

## Dataset and Data

Compare datasets, populations, sample characteristics and
preprocessing when available.

## Models and Approaches

Compare the main models, algorithms and technical approaches.

## Results and Findings

Compare important reported findings and evaluation metrics.

## Contributions

Compare the main contributions and novelty.

## Limitations

Compare the supported limitations.

## Key Similarities

Give the most important common characteristics.

## Key Differences

Give the most important differences.

## Overall Assessment

Give a concise academic synthesis explaining what the comparison
means for research.

============================================================
FOCUSED QUESTIONS
============================================================

If the researcher asks a focused question, DO NOT generate every
section above.

Instead, directly answer the question.

For example:

If the question is about methodology:
focus on methodology, models, experimental design and relevant
results.

If the question is about datasets:
focus on datasets, population, size, source, preprocessing and
their implications.

If the question is about results:
focus on metrics, experiments, findings and evaluation.

If the question is about limitations:
focus on reported and evidence-supported limitations.

If the question is about research gaps:
focus on unresolved problems and cross-paper gaps.

If the question is about choosing a paper:
compare the papers against the requested criterion and explain
the evidence.

============================================================
MULTI-PAPER RULE
============================================================

For 2 papers:
make direct Paper 1 vs Paper 2 comparisons.

For 3–5 papers:
compare each paper and then synthesize common patterns.

For 6–10 papers:
avoid repeating every detail.

Instead identify:
- dominant patterns
- major outliers
- methodological clusters
- important differences
- strongest findings
- recurring limitations

Still ensure every paper contributes to the synthesis.

============================================================
FINAL QUALITY CHECK
============================================================

Before producing the answer, internally verify:

- Did I consider every selected paper?
- Did I answer the actual researcher question?
- Did I distinguish evidence from interpretation?
- Did I avoid invented information?
- Did I compare rather than merely summarize?
- Did I keep the response concise?
- Did I preserve important numerical results?
- Did I explain the academic meaning of major differences?

Return ONLY the final academic comparison.

============================================================
PAPER EVIDENCE
============================================================

{evidence_context}
"""

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
    ) -> Optional[str]:

        try:

            print(
                "Paper Comparison | "
                "Calling academic synthesis model..."
            )

            response = (
                chat_service._call_llm(
                    prompt,
                    max_tokens=self.LLM_MAX_TOKENS,
                )
            )

            if isinstance(
                response,
                str,
            ):

                return response.strip()

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

                        value = value.strip()

                        if value:
                            return value

            return None

        except Exception as exc:

            print(
                "Paper Comparison | "
                f"LLM generation failed: {exc}"
            )

            return None

    # ============================================================
    # CLEAN LLM RESPONSE
    # ============================================================

    def _clean_llm_response(
        self,
        answer: str,
    ) -> str:

        if not answer:
            return ""

        text = str(
            answer
        ).strip()

        # --------------------------------------------------------
        # Remove <think> blocks.
        # --------------------------------------------------------

        while True:

            lower = text.lower()

            start = lower.find(
                "<think>"
            )

            if start == -1:
                break

            end = lower.find(
                "</think>",
                start,
            )

            if end == -1:

                text = text[
                    :start
                ].strip()

                break

            text = (
                text[:start]
                + text[
                    end + len("</think>"):
                ]
            ).strip()

        # --------------------------------------------------------
        # Remove common model prefixes.
        # --------------------------------------------------------

        prefixes = [
            "final answer:",
            "final response:",
            "answer:",
            "response:",
        ]

        lower = text.lower()

        for prefix in prefixes:

            if lower.startswith(
                prefix
            ):

                text = text[
                    len(prefix):
                ].strip()

                lower = text.lower()

        # --------------------------------------------------------
        # HTML cleanup.
        # --------------------------------------------------------

        text = (
            text
            .replace(
                "<br>",
                "\n",
            )
            .replace(
                "<br/>",
                "\n",
            )
            .replace(
                "<br />",
                "\n",
            )
        )

        # --------------------------------------------------------
        # Remove accidental code fences.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Remove excessive blank lines.
        # --------------------------------------------------------

        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        # --------------------------------------------------------
        # Remove internal-system leakage.
        # --------------------------------------------------------

        forbidden_lines = []

        for line in text.splitlines():

            lower_line = line.lower()

            if any(
                phrase in lower_line
                for phrase in (
                    "as an ai language model",
                    "as an ai assistant",
                    "my prompt is",
                    "the prompt says",
                    "retrieved chunks",
                    "vector database",
                    "qdrant",
                    "embedding model",
                    "internal system",
                )
            ):

                continue

            forbidden_lines.append(
                line
            )

        text = "\n".join(
            forbidden_lines
        )

        return text.strip()

    # ============================================================
    # FALLBACK COMPARISON
    # ============================================================

    def _fallback_comparison(
        self,
        paper_ids: List[int],
        evidence: Dict[int, Dict[str, Any]],
        question: str,
    ) -> str:

        sections = [
            "## Comparison",
            "",
            (
                "The language-generation service was unavailable, "
                "so the following comparison is based directly on "
                "the available evidence from the selected papers."
            ),
        ]

        if question:
            sections.extend(
                [
                    "",
                    "### Researcher Question",
                    "",
                    question,
                ]
            )

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            paper_data = evidence.get(
                paper_id,
                {},
            )

            paper_name = paper_data.get(
                "paper_name",
                f"Paper {paper_id}",
            )

            sections.extend(
                [
                    "",
                    f"### Paper {index}: {paper_name}",
                ]
            )

            categories = paper_data.get(
                "categories",
                {},
            )

            for category_name, _ in self.CATEGORIES:

                chunks = categories.get(
                    category_name,
                    [],
                )

                if not chunks:
                    continue

                sections.append(
                    f"\n**{category_name}:**"
                )

                for item in chunks[:2]:

                    text = item.get(
                        "text",
                        "",
                    )

                    if text:
                        sections.append(
                            text[:900]
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

        seen = set()

        for paper_id in paper_ids:

            paper_data = evidence.get(
                paper_id,
                {},
            )

            paper_name = paper_data.get(
                "paper_name",
                f"Paper {paper_id}",
            )

            chunks = paper_data.get(
                "all_chunks",
                [],
            )

            for item in chunks:

                text = item.get(
                    "text",
                    "",
                )

                if not text:
                    continue

                fingerprint = (
                    paper_id,
                    self._compact_text(
                        text
                    )[:800],
                )

                if fingerprint in seen:
                    continue

                seen.add(
                    fingerprint
                )

                sources.append(
                    {
                        "paper_id": paper_id,
                        "paper_name": paper_name,
                        "category": self._find_source_category(
                            paper_id,
                            text,
                            evidence,
                        ),
                        "text": text,
                        "score": item.get(
                            "score",
                            0.0,
                        ),
                    }
                )

        return sources

    # ============================================================
    # FIND SOURCE CATEGORY
    # ============================================================

    def _find_source_category(
        self,
        paper_id: int,
        text: str,
        evidence: Dict[int, Dict[str, Any]],
    ) -> str:

        normalized = self._compact_text(
            text
        ).lower()

        categories = (
            evidence
            .get(
                paper_id,
                {},
            )
            .get(
                "categories",
                {},
            )
        )

        for category_name, chunks in (
            categories.items()
        ):

            for chunk in chunks:

                chunk_text = (
                    self._compact_text(
                        chunk.get(
                            "text",
                            "",
                        )
                    )
                    .lower()
                )

                if (
                    chunk_text
                    and (
                        chunk_text[:200]
                        in normalized
                        or normalized[:200]
                        in chunk_text
                    )
                ):

                    return category_name

        return "Research Evidence"

    # ============================================================
    # MAIN COMPARE METHOD
    # ============================================================

    def compare(
        self,
        paper_ids: List[int],
        evidence_per_paper: int = 5,
        user_question: str = "",
    ) -> Dict[str, Any]:

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        if not paper_ids:

            raise ValueError(
                "At least two papers are required."
            )

        normalized_ids = []

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

            if (
                value > 0
                and value not in normalized_ids
            ):

                normalized_ids.append(
                    value
                )

        if len(normalized_ids) < self.MIN_PAPERS:

            raise ValueError(
                "Paper comparison requires "
                "at least two different papers."
            )

        if len(normalized_ids) > self.MAX_PAPERS:

            raise ValueError(
                "Paper comparison supports "
                "a maximum of 10 papers."
            )

        # --------------------------------------------------------
        # NORMALIZE QUESTION
        # --------------------------------------------------------

        question = (
            str(
                user_question
            ).strip()
            if user_question
            else ""
        )

        if not question:

            question = (
                "Compare the selected research papers. "
                "Explain their major similarities and differences, "
                "including objectives, methodology, datasets, "
                "models, results, contributions, limitations, "
                "and implications for future research."
            )

        print(
            "=================================================="
        )

        print(
            "Paper Comparison | "
            f"{len(normalized_ids)} papers"
        )

        print(
            "Paper Comparison | "
            f"Question: {question}"
        )

        # --------------------------------------------------------
        # QUESTION INTENT
        # --------------------------------------------------------

        intent = (
            self._detect_question_intent(
                question
            )
        )

        # --------------------------------------------------------
        # RETRIEVE
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
        # GENERATE
        # --------------------------------------------------------

        prompt = (
            self._build_prompt(
                paper_ids=normalized_ids,
                evidence=evidence,
                question=question,
                intent=intent,
            )
        )

        answer = self._call_llm(
            prompt
        )

        # --------------------------------------------------------
        # CLEAN
        # --------------------------------------------------------

        if answer:

            answer = (
                self._clean_llm_response(
                    answer
                )
            )

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        generation_status = (
            "ai_generated"
        )

        if not answer:

            generation_status = (
                "evidence_fallback"
            )

            answer = (
                self._fallback_comparison(
                    paper_ids=normalized_ids,
                    evidence=evidence,
                    question=question,
                )
            )

        # --------------------------------------------------------
        # SOURCES
        # --------------------------------------------------------

        sources = (
            self._build_sources(
                paper_ids=normalized_ids,
                evidence=evidence,
            )
        )

        # --------------------------------------------------------
        # PAPER INFORMATION
        # --------------------------------------------------------

        papers = []

        for paper_id in normalized_ids:

            paper_data = evidence.get(
                paper_id,
                {},
            )

            papers.append(
                {
                    "paper_id": paper_id,
                    "paper_name": paper_data.get(
                        "paper_name",
                        f"Paper {paper_id}",
                    ),
                }
            )

        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        result = {
            "paper_ids": normalized_ids,
            "papers_count": len(
                normalized_ids
            ),
            "papers": papers,
            "question": question,
            "comparison": answer,
            "sources": sources,
            "generation_status": generation_status,
            "analysis_type": intent,
        }

        print(
            "Paper Comparison | "
            f"Completed | status={generation_status}"
        )

        print(
            "=================================================="
        )

        return result


# ================================================================
# SINGLETON
# IMPORTANT:
# routes.py imports this exact object.
# ================================================================

comparison_service = ComparisonService()