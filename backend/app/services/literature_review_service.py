from typing import Dict, List, Any
import re

from app.services.multi_document_service import multi_document_service
from app.services.chat_service import chat_service


class LiteratureReviewService:
    """
    PaperAxiom evidence-grounded literature review service.

    Main goals:
        - Retrieve evidence from selected papers.
        - Keep each paper clearly separated.
        - Remove duplicate evidence.
        - Give the LLM enough context to understand the papers.
        - Generate a concise academic synthesis.
        - Compare studies instead of merely listing them.
        - Never invent unsupported paper information.
        - Preserve the existing API/service interface.
    """

    # ============================================================
    # RETRIEVAL CATEGORIES
    # ============================================================

    CATEGORIES = [
        (
            "Research Focus",
            (
                "research problem, research question, objective, "
                "aim, motivation, hypothesis, clinical or scientific "
                "problem, study purpose"
            ),
        ),
        (
            "Methodology",
            (
                "methodology, research design, experimental setup, "
                "workflow, pipeline, architecture, procedure, "
                "training strategy, validation strategy"
            ),
        ),
        (
            "Dataset and Data",
            (
                "dataset, data source, sample size, population, "
                "patients, imaging data, annotations, labels, "
                "train validation test split, modality"
            ),
        ),
        (
            "Models and Techniques",
            (
                "model, algorithm, deep learning, machine learning, "
                "neural network, architecture, feature extraction, "
                "loss function, optimization, preprocessing, "
                "classification, segmentation, detection"
            ),
        ),
        (
            "Results and Findings",
            (
                "results, findings, performance, accuracy, precision, "
                "recall, F1, AUC, Dice, IoU, sensitivity, specificity, "
                "evaluation, comparison, quantitative results"
            ),
        ),
        (
            "Contribution and Novelty",
            (
                "contribution, novelty, innovation, proposed method, "
                "advancement, significance, improvement, main contribution"
            ),
        ),
        (
            "Limitations",
            (
                "limitations, weaknesses, constraints, challenges, "
                "failure cases, generalization problems, limitations "
                "of dataset or methodology"
            ),
        ),
        (
            "Future Work",
            (
                "future work, future research, recommendations, "
                "extensions, improvements, unresolved problems"
            ),
        ),
    ]

    # ============================================================
    # BASIC TEXT HELPERS
    # ============================================================

    def _clean_text(self, text: Any) -> str:
        """Normalize whitespace without changing meaning."""

        if text is None:
            return ""

        return " ".join(
            str(text)
            .replace("\r", " ")
            .replace("\n", " ")
            .split()
        ).strip()

    def _normalize(self, text: str) -> str:
        """Normalize text for duplicate comparison."""

        return self._clean_text(text).lower()

    # ============================================================
    # DUPLICATE DETECTION
    # ============================================================

    def _is_duplicate(
        self,
        text: str,
        existing: List[str],
    ) -> bool:
        """
        Detect exact and highly similar retrieved chunks.

        This prevents the same paragraph from consuming the
        LLM context multiple times.
        """

        normalized = self._normalize(text)

        if not normalized:
            return True

        words_a = set(
            normalized.split()
        )

        for old in existing:

            old_normalized = self._normalize(
                old
            )

            if normalized == old_normalized:
                return True

            words_b = set(
                old_normalized.split()
            )

            if (
                len(words_a) < 20
                or len(words_b) < 20
            ):
                continue

            union = words_a | words_b

            if not union:
                continue

            similarity = (
                len(words_a & words_b)
                / len(union)
            )

            if similarity >= 0.88:
                return True

        return False

    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Retrieve evidence separately for every selected paper.

        Each category retrieves focused evidence so that the LLM
        receives information about the paper's problem, method,
        data, results, contribution, limitations and future work.
        """

        evidence_by_paper = {
            paper_id: []
            for paper_id in paper_ids
        }

        for category, query in self.CATEGORIES:

            print(
                f"Literature review | Retrieving: {category}"
            )

            try:

                result = (
                    multi_document_service.search(
                        query=query,
                        paper_ids=paper_ids,
                        limit_per_paper=2,
                    )
                )

            except Exception as exc:

                print(
                    "Literature review retrieval failed | "
                    f"{category} | {exc}"
                )

                continue

            if not isinstance(
                result,
                dict,
            ):
                continue

            results = result.get(
                "results",
                [],
            )

            if not isinstance(
                results,
                list,
            ):
                continue

            for item in results:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                paper_id = item.get(
                    "paper_id"
                )

                if paper_id not in evidence_by_paper:
                    continue

                text = self._clean_text(
                    item.get(
                        "text",
                        "",
                    )
                )

                if not text:
                    continue

                existing = [
                    x["text"]
                    for x in evidence_by_paper[
                        paper_id
                    ]
                ]

                if self._is_duplicate(
                    text,
                    existing,
                ):
                    continue

                try:

                    score = float(
                        item.get(
                            "score",
                            0.0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    score = 0.0

                evidence_by_paper[
                    paper_id
                ].append(
                    {
                        "category": category,
                        "text": text[:1000],
                        "score": score,
                    }
                )

        # ========================================================
        # Keep strongest evidence.
        #
        # We intentionally keep evidence from different categories
        # rather than simply taking the globally highest scores.
        # ========================================================

        for paper_id in evidence_by_paper:

            items = evidence_by_paper[
                paper_id
            ]

            selected = []

            # First take the strongest item from each category.
            for category, _ in self.CATEGORIES:

                category_items = [
                    item
                    for item in items
                    if item["category"] == category
                ]

                category_items.sort(
                    key=lambda x: x["score"],
                    reverse=True,
                )

                if category_items:
                    selected.append(
                        category_items[0]
                    )

            # Then fill remaining slots with strongest evidence.
            remaining = [
                item
                for item in items
                if item not in selected
            ]

            remaining.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            selected.extend(
                remaining[:5]
            )

            evidence_by_paper[
                paper_id
            ] = selected[:12]

            print(
                "Literature review evidence | "
                f"Paper {paper_id} | "
                f"Chunks: "
                f"{len(evidence_by_paper[paper_id])}"
            )

        return evidence_by_paper

    # ============================================================
    # BUILD STRUCTURED CONTEXT
    # ============================================================

    def _build_context(
        self,
        paper_ids: List[int],
        evidence_by_paper: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """
        Build a clearly separated context for the LLM.

        Important:
        The model must never confuse evidence from Paper A
        with evidence from Paper B.
        """

        sections = []

        for index, paper_id in enumerate(
            paper_ids,
            start=1,
        ):

            sections.append(
                "\n"
                + "=" * 80
                + f"\nPAPER {index} | PAPER ID: {paper_id}\n"
                + "=" * 80
            )

            evidence = evidence_by_paper.get(
                paper_id,
                [],
            )

            if not evidence:

                sections.append(
                    "No relevant evidence was retrieved "
                    "for this paper."
                )

                continue

            for category, _ in self.CATEGORIES:

                category_items = [
                    item
                    for item in evidence
                    if item["category"] == category
                ]

                if not category_items:
                    continue

                sections.append(
                    f"\n[{category}]"
                )

                for item in category_items:

                    sections.append(
                        f"\n- {item['text']}"
                    )

        return "\n".join(
            sections
        )

    # ============================================================
    # FALLBACK
    # ============================================================

    def _build_fallback_review(
        self,
        paper_ids: List[int],
        evidence_by_paper: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """
        Safe fallback when the LLM is unavailable.

        It does not invent content.
        """

        parts = [
            "# Literature Review\n\n",
            (
                "The selected studies were reviewed using the "
                "available evidence from each paper. The synthesis "
                "below is limited to information supported by the "
                "retrieved research material.\n\n"
            ),
        ]

        # --------------------------------------------------------
        # Research focus
        # --------------------------------------------------------

        parts.append(
            "## Research Focus\n\n"
        )

        for paper_id in paper_ids:

            items = [
                item
                for item in evidence_by_paper.get(
                    paper_id,
                    [],
                )
                if item["category"]
                == "Research Focus"
            ]

            if items:

                parts.append(
                    f"**Paper {paper_id}:** "
                    f"{items[0]['text']}\n\n"
                )

        # --------------------------------------------------------
        # Methodology
        # --------------------------------------------------------

        parts.append(
            "## Methodological Approaches\n\n"
        )

        for paper_id in paper_ids:

            items = [
                item
                for item in evidence_by_paper.get(
                    paper_id,
                    [],
                )
                if item["category"]
                == "Methodology"
            ]

            if items:

                parts.append(
                    f"**Paper {paper_id}:** "
                    f"{items[0]['text']}\n\n"
                )

        # --------------------------------------------------------
        # Results
        # --------------------------------------------------------

        parts.append(
            "## Results and Findings\n\n"
        )

        for paper_id in paper_ids:

            items = [
                item
                for item in evidence_by_paper.get(
                    paper_id,
                    [],
                )
                if item["category"]
                == "Results and Findings"
            ]

            if items:

                parts.append(
                    f"**Paper {paper_id}:** "
                    f"{items[0]['text']}\n\n"
                )

        # --------------------------------------------------------
        # Contributions
        # --------------------------------------------------------

        parts.append(
            "## Contributions\n\n"
        )

        for paper_id in paper_ids:

            items = [
                item
                for item in evidence_by_paper.get(
                    paper_id,
                    [],
                )
                if item["category"]
                == "Contribution and Novelty"
            ]

            if items:

                parts.append(
                    f"**Paper {paper_id}:** "
                    f"{items[0]['text']}\n\n"
                )

        # --------------------------------------------------------
        # Limitations
        # --------------------------------------------------------

        parts.append(
            "## Limitations and Future Directions\n\n"
        )

        for paper_id in paper_ids:

            limitations = [
                item
                for item in evidence_by_paper.get(
                    paper_id,
                    [],
                )
                if item["category"]
                == "Limitations"
            ]

            future = [
                item
                for item in evidence_by_paper.get(
                    paper_id,
                    [],
                )
                if item["category"]
                == "Future Work"
            ]

            if limitations:

                parts.append(
                    f"**Paper {paper_id} limitations:** "
                    f"{limitations[0]['text']}\n\n"
                )

            if future:

                parts.append(
                    f"**Paper {paper_id} future work:** "
                    f"{future[0]['text']}\n\n"
                )

        parts.append(
            (
                "## Comparative Assessment\n\n"
                "The available evidence shows that the selected "
                "studies address related research questions while "
                "differing in methodology, data, computational "
                "techniques, and reported findings. A definitive "
                "cross-paper research gap should only be stated "
                "when it is directly supported by the evidence."
            )
        )

        return "".join(
            parts
        )

    # ============================================================
    # CLEAN LLM OUTPUT
    # ============================================================

    def _clean_llm_output(
        self,
        answer: Any,
    ) -> str:
        """
        Remove accidental code fences and unnecessary whitespace.
        """

        if answer is None:
            return ""

        text = str(
            answer
        ).strip()

        text = re.sub(
            r"^```(?:markdown|md|text)?\s*",
            "",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        return text.strip()

    # ============================================================
    # GENERATION PROMPT
    # ============================================================

    def _build_prompt(
        self,
        paper_ids: List[int],
        evidence_context: str,
    ) -> str:
        """
        Strong academic synthesis prompt.

        The important change from the old version is that the LLM
        is explicitly instructed to synthesize relationships between
        papers rather than produce repetitive paper-by-paper summaries.
        """

        paper_labels = ", ".join(
            f"Paper {paper_id}"
            for paper_id in paper_ids
        )

        return f"""
You are PaperAxiom, an expert academic research assistant.

Your task is to write a concise, high-quality literature review
from the supplied research-paper evidence.

SELECTED PAPERS:
{paper_labels}

============================================================
SOURCE MATERIAL
============================================================

{evidence_context}

============================================================
CORE RULE
============================================================

The supplied papers are the PRIMARY source of truth.

Use the evidence to understand the papers deeply before writing.

Do not invent:
- authors
- paper titles
- datasets
- numerical results
- methods
- claims
- limitations
- research gaps
- citations

If information is not supported by the supplied evidence,
do not manufacture it.

============================================================
WHAT A GOOD REVIEW SHOULD DO
============================================================

Do NOT write:

"Paper 1 did X.
Paper 2 did Y.
Paper 3 did Z."

That is a paper summary list.

Instead, synthesize the studies.

For example:

"Recent studies have increasingly explored X through
deep-learning-based approaches. While one study focuses on A,
another extends the problem toward B, resulting in differences
in both methodology and evaluation. Collectively, the findings
suggest..., although differences in datasets and experimental
settings limit direct comparison."

The final review should therefore connect the studies.

============================================================
PAPER IDENTIFICATION
============================================================

When discussing an individual study, identify it clearly as:

- Paper 1
- Paper 2
- Paper 3

If the evidence contains a reliable paper title, use the title
naturally as well.

Never invent a title.

============================================================
REQUIRED CONTENT
============================================================

1. Research Landscape

Start with one concise paragraph explaining:

- the overall research area
- the common problem addressed
- why the problem matters
- the general direction represented by the selected studies

2. Research Focus

Explain how the research objectives differ or overlap.

Identify common themes and important differences.

3. Methodological Synthesis

Compare:

- research approaches
- architectures
- algorithms
- experimental designs
- preprocessing
- training strategies
- validation approaches

Focus on meaningful differences rather than listing everything.

4. Data and Experimental Settings

Compare datasets and data characteristics when supported.

Mention:

- dataset type
- modality
- sample size
- patient/population characteristics
- annotations
- train/test design

Only include details supported by evidence.

5. Findings

Synthesize the major findings.

If numerical metrics are available, preserve them accurately.

Explain what the findings mean rather than merely listing numbers.

6. Contributions

Explain how the papers contribute to the field.

Highlight complementary contributions where appropriate.

7. Limitations

Discuss limitations explicitly reported by the studies.

Do not invent limitations.

8. Comparative Synthesis

This is one of the most important sections.

Clearly explain:

- what the papers have in common
- how their methods differ
- how their datasets differ
- how their findings compare
- which approaches appear complementary
- where direct comparison is difficult

9. Research Gap

Only identify a research gap if the combined evidence supports it.

A valid gap may involve:

- an unresolved problem
- a limitation shared across studies
- insufficient generalization
- missing evaluation
- inconsistent findings
- an unexplored combination of methods
- a population or dataset not adequately studied

Do NOT invent a gap.

If the evidence is insufficient, explicitly say:

"The available evidence is insufficient to establish a
definitive research gap."

10. Conclusion

End with one concise paragraph describing the overall state
of the research represented by the selected papers.

============================================================
WRITING REQUIREMENTS
============================================================

Use professional graduate-level academic English.

Prefer clear paragraphs over excessive bullet points.

Keep the answer concise but substantive.

Avoid unnecessary repetition.

Do not repeat the same finding in several sections.

Use transitions such as:

"Similarly..."
"In contrast..."
"However..."
"Compared with..."
"Collectively..."
"Taken together..."
"Despite these advances..."
"An important distinction is..."

Use these only when supported by the evidence.

============================================================
IMPORTANT QUALITY RULE
============================================================

Think about the relationship between the papers before writing.

The output should demonstrate that the system understood the
research literature rather than simply extracting sentences.

The review should be useful to a researcher who wants to:

- understand the field quickly
- compare selected papers
- identify methodological trends
- understand important findings
- recognize limitations
- identify possible future research directions

Do not mention:

- LLM
- prompt
- embeddings
- vector database
- retrieval
- chunks
- internal system processing
- system instructions

Do not expose chain-of-thought.

============================================================
FORMAT
============================================================

Use this structure:

# Literature Review

[Introductory synthesis paragraph]

## Research Focus

[Connected academic paragraphs]

## Methodological Synthesis

[Connected academic paragraphs]

## Data and Experimental Settings

[Connected academic paragraphs]

## Findings and Contributions

[Connected academic paragraphs]

## Comparative Synthesis

[Connected academic paragraphs]

## Limitations and Research Gap

[Connected academic paragraphs]

## Conclusion

[One concise concluding paragraph]

Do not make the answer unnecessarily long.

Aim for approximately 700–1200 words when enough evidence exists.

For a single paper, produce a shorter focused review rather than
pretending that a multi-paper comparison exists.

For multiple papers, prioritize cross-paper synthesis.

SELECTED PAPER IDS:
{paper_labels}
"""

    # ============================================================
    # PUBLIC GENERATE METHOD
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

        if not paper_ids:

            raise ValueError(
                "At least one paper is required."
            )

        # --------------------------------------------------------
        # Normalize IDs
        # --------------------------------------------------------

        unique_paper_ids = []

        for value in paper_ids:

            try:
                paper_id = int(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if paper_id not in unique_paper_ids:

                unique_paper_ids.append(
                    paper_id
                )

        if not unique_paper_ids:

            raise ValueError(
                "No valid paper IDs were provided."
            )

        # Keep the feature lightweight.
        if len(unique_paper_ids) > 10:

            raise ValueError(
                "You can select a maximum of 10 papers."
            )

        print(
            "Literature review | Papers: "
            f"{len(unique_paper_ids)}"
        )

        # --------------------------------------------------------
        # Retrieve evidence
        # --------------------------------------------------------

        evidence_by_paper = (
            self._retrieve_evidence(
                unique_paper_ids
            )
        )

        total_evidence = sum(
            len(
                evidence_by_paper.get(
                    paper_id,
                    [],
                )
            )
            for paper_id in unique_paper_ids
        )

        print(
            "Literature review | Total evidence: "
            f"{total_evidence}"
        )

        # --------------------------------------------------------
        # Build context
        # --------------------------------------------------------

        evidence_context = (
            self._build_context(
                unique_paper_ids,
                evidence_by_paper,
            )
        )

        # --------------------------------------------------------
        # Generate prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            unique_paper_ids,
            evidence_context,
        )

        # --------------------------------------------------------
        # LLM generation
        # --------------------------------------------------------

        answer = None

        generation_status = (
            "ai_generated"
        )

        try:

            print(
                "Literature review | "
                "Generating academic synthesis..."
            )

            answer = chat_service._call_llm(
                prompt,
                max_tokens=1800,
            )

        except Exception as exc:

            print(
                "Literature review | "
                f"Generation failed: {exc}"
            )

        # --------------------------------------------------------
        # Retry with smaller response
        # --------------------------------------------------------

        if not answer:

            try:

                print(
                    "Literature review | "
                    "Retrying with compact generation..."
                )

                compact_prompt = (
                    prompt
                    + """

IMPORTANT:
Generate a more concise version.
Prioritize comparative synthesis, findings,
limitations, and research gap.
"""
                )

                answer = chat_service._call_llm(
                    compact_prompt,
                    max_tokens=1100,
                )

            except Exception as exc:

                print(
                    "Literature review | "
                    f"Retry failed: {exc}"
                )

        # --------------------------------------------------------
        # Fallback
        # --------------------------------------------------------

        if not answer:

            print(
                "Literature review | "
                "Using evidence-grounded fallback."
            )

            generation_status = (
                "evidence_fallback"
            )

            answer = self._build_fallback_review(
                unique_paper_ids,
                evidence_by_paper,
            )

        # --------------------------------------------------------
        # Clean response
        # --------------------------------------------------------

        answer = self._clean_llm_output(
            answer
        )

        if not answer:

            generation_status = (
                "evidence_fallback"
            )

            answer = self._build_fallback_review(
                unique_paper_ids,
                evidence_by_paper,
            )

        # --------------------------------------------------------
        # Sources for frontend
        # --------------------------------------------------------

        sources = []

        for paper_id in unique_paper_ids:

            for item in evidence_by_paper.get(
                paper_id,
                [],
            ):

                sources.append(
                    {
                        "paper_id": paper_id,
                        "category": item.get(
                            "category",
                            "",
                        ),
                        "text": item.get(
                            "text",
                            "",
                        ),
                        "score": item.get(
                            "score",
                            0.0,
                        ),
                    }
                )

        # --------------------------------------------------------
        # Final response
        # --------------------------------------------------------

        return {
            "paper_ids": unique_paper_ids,
            "papers_count": len(
                unique_paper_ids
            ),
            "review": answer,
            "sources": sources,
            "source_count": len(
                sources
            ),
            "generation_status": generation_status,
        }


# ================================================================
# GLOBAL SERVICE INSTANCE
# ================================================================

literature_review_service = (
    LiteratureReviewService()
)