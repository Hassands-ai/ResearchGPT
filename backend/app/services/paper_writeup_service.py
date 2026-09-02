from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.services.multi_document_service import (
    multi_document_service,
)
from app.services.chat_service import chat_service


class PaperWriteupService:
    """
    ResearchGPT Paper Write-up Service.

    Generates concise academic sections from selected uploaded
    research papers.

    Primary source:
        Uploaded papers / retrieved evidence

    Generation:
        Existing ResearchGPT ChatService / configured LLM

    Supported write-up types:
        - abstract
        - introduction
        - related_work
        - methodology
        - results_discussion
        - conclusion
        - full_paper

    Design goals:
        - simple
        - fast
        - evidence-grounded
        - reusable
        - compatible with existing ResearchGPT architecture
    """

    MAX_PAPERS = 10
    EVIDENCE_PER_PAPER = 5
    MAX_EVIDENCE_CHARS = 18000

    WRITEUP_TYPES = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "related_work": "Related Work / Literature Review",
        "methodology": "Methodology",
        "results_discussion": "Results and Discussion",
        "conclusion": "Conclusion",
        "full_paper": "Full Paper Draft",
    }

    def __init__(self) -> None:
        self.multi_document_service = (
            multi_document_service
        )
        self.chat_service = chat_service

    # ============================================================
    # PUBLIC METHOD
    # ============================================================

    def generate(
        self,
        paper_ids: List[int],
        writeup_type: str = "introduction",
        research_topic: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an academic write-up from selected papers.

        Returns a stable dictionary suitable for the API/frontend.
        """

        paper_ids = self._normalize_paper_ids(
            paper_ids
        )

        if not paper_ids:
            raise ValueError(
                "Select at least one paper."
            )

        if len(paper_ids) > self.MAX_PAPERS:
            raise ValueError(
                "You can select a maximum of "
                f"{self.MAX_PAPERS} papers."
            )

        writeup_type = (
            self._normalize_writeup_type(
                writeup_type
            )
        )

        evidence = self._retrieve_evidence(
            paper_ids=paper_ids,
            research_topic=research_topic,
        )

        evidence_text = (
            self._format_evidence(
                evidence
            )
        )

        prompt = self._build_prompt(
            paper_ids=paper_ids,
            writeup_type=writeup_type,
            research_topic=research_topic,
            instructions=instructions,
            evidence_text=evidence_text,
        )

        answer = self._call_llm(
            prompt
        )

        if not answer:
            answer = self._fallback_writeup(
                writeup_type=writeup_type,
                evidence=evidence,
            )

        answer = self._clean_output(
            answer
        )

        return {
            "generation_status": (
                "ai_generated"
                if answer
                else "fallback"
            ),
            "writeup_type": writeup_type,
            "writeup_type_label": self.WRITEUP_TYPES[
                writeup_type
            ],
            "papers_count": len(
                paper_ids
            ),
            "paper_ids": paper_ids,
            "source_count": len(
                evidence
            ),
            "research_topic": (
                research_topic or ""
            ),
            "content": answer,
            "writeup": answer,
            "evidence": evidence,
        }

    # ============================================================
    # NORMALIZATION
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

    def _normalize_writeup_type(
        self,
        writeup_type: str,
    ) -> str:

        value = (
            str(
                writeup_type or ""
            )
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "abstract": "abstract",
            "intro": "introduction",
            "introduction": "introduction",
            "related": "related_work",
            "related_work": "related_work",
            "literature": "related_work",
            "literature_review": "related_work",
            "method": "methodology",
            "methods": "methodology",
            "methodology": "methodology",
            "results": "results_discussion",
            "discussion": "results_discussion",
            "results_discussion": "results_discussion",
            "conclusion": "conclusion",
            "full": "full_paper",
            "full_paper": "full_paper",
            "paper": "full_paper",
        }

        normalized = aliases.get(
            value,
            "introduction",
        )

        return normalized

    # ============================================================
    # EVIDENCE RETRIEVAL
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
        research_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        queries = self._build_queries(
            research_topic
        )

        collected = []

        for paper_id in paper_ids:

            for query in queries:

                results = (
                    self._retrieve(
                        query=query,
                        paper_ids=[
                            paper_id
                        ],
                    )
                )

                for result in results:

                    text = self._extract_text(
                        result
                    )

                    if not text:
                        continue

                    item = dict(
                        result
                    )

                    item["paper_id"] = (
                        paper_id
                    )

                    item["text"] = text

                    collected.append(
                        item
                    )

        return self._deduplicate(
            collected
        )[
            : self.MAX_EVIDENCE_CHARS
        ]

    def _build_queries(
        self,
        research_topic: Optional[str],
    ) -> List[str]:

        base = (
            research_topic.strip()
            if research_topic
            else "research study"
        )

        return [
            (
                f"{base} research problem "
                "objective motivation"
            ),
            (
                f"{base} methodology "
                "dataset model experimental setup"
            ),
            (
                f"{base} results findings "
                "evaluation performance"
            ),
            (
                f"{base} contribution "
                "limitation future work"
            ),
        ]

    def _retrieve(
        self,
        query: str,
        paper_ids: List[int],
    ) -> List[Dict[str, Any]]:

        service = (
            self.multi_document_service
        )

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
                    "limit_per_paper": (
                        self.EVIDENCE_PER_PAPER
                    ),
                },
                {
                    "query": query,
                    "paper_ids": paper_ids,
                    "limit": (
                        self.EVIDENCE_PER_PAPER
                    ),
                },
                {
                    "query": query,
                    "paper_ids": paper_ids,
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
                    continue

                except Exception as exc:

                    print(
                        "Paper Write-up retrieval warning: "
                        f"{method_name}: {exc}"
                    )

                    break

        return []

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

        return []

    # ============================================================
    # EVIDENCE HELPERS
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
        ):

            value = item.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                value = (
                    value
                    .replace(
                        "\x00",
                        " ",
                    )
                    .strip()
                )

                if value:
                    return value

        return ""

    def _deduplicate(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        result = []
        seen = set()

        for item in items:

            text = self._extract_text(
                item
            )

            if not text:
                continue

            fingerprint = (
                " ".join(
                    text.lower().split()
                )[:800]
            )

            if fingerprint in seen:
                continue

            seen.add(
                fingerprint
            )

            copied = dict(
                item
            )

            copied["text"] = text

            result.append(
                copied
            )

        return result

    def _format_evidence(
        self,
        evidence: List[Dict[str, Any]],
    ) -> str:

        if not evidence:
            return (
                "No retrieved evidence is available."
            )

        sections = []
        total_chars = 0

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            paper_id = item.get(
                "paper_id",
                "Unknown",
            )

            text = self._extract_text(
                item
            )

            if not text:
                continue

            text = text[:3000]

            block = (
                f"\n--- Evidence {index} "
                f"(Paper {paper_id}) ---\n"
                f"{text}"
            )

            if (
                total_chars
                + len(block)
                > self.MAX_EVIDENCE_CHARS
            ):
                break

            sections.append(
                block
            )

            total_chars += len(
                block
            )

        return "\n".join(
            sections
        )

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        paper_ids: List[int],
        writeup_type: str,
        research_topic: Optional[str],
        instructions: Optional[str],
        evidence_text: str,
    ) -> str:

        section = self.WRITEUP_TYPES[
            writeup_type
        ]

        topic = (
            research_topic.strip()
            if research_topic
            else "Not explicitly specified"
        )

        extra = (
            instructions.strip()
            if instructions
            else "None"
        )

        return f"""
You are ResearchGPT, an academic research-writing assistant.

Generate a high-quality academic {section} using the supplied
evidence from uploaded research papers.

SELECTED PAPERS:
{", ".join(str(x) for x in paper_ids)}

RESEARCH TOPIC:
{topic}

ADDITIONAL INSTRUCTIONS:
{extra}

==================================================
SUPPLIED PAPER EVIDENCE
==================================================

{evidence_text}

==================================================
WRITING REQUIREMENTS
==================================================

1. The uploaded papers are the primary source of truth.

2. Understand the evidence before writing.

3. Use the supplied papers to build a coherent academic narrative.

4. You may use general academic knowledge to improve wording,
   structure, and explanation, but do NOT introduce unsupported
   scientific claims.

5. Never invent:
   - experimental results
   - datasets
   - sample sizes
   - model names
   - numerical metrics
   - clinical findings
   - citations
   - authors
   - publication details

6. If an important fact is not supported by the evidence,
   write around it rather than inventing it.

7. Keep the writing concise and information-dense.

8. Prefer connected academic paragraphs over excessive bullets.

9. Avoid repetitive statements.

10. Use professional graduate-level academic English.

11. Maintain logical transitions between paragraphs.

12. Do not mention:
    - LLM
    - prompt
    - retrieval
    - chunks
    - vector database
    - internal system
    - AI generation process

13. Do not claim that the supplied papers prove something
    they do not actually establish.

==================================================
SECTION-SPECIFIC REQUIREMENTS
==================================================
"""

        + self._section_instructions(
            writeup_type
        )

    def _section_instructions(
        self,
        writeup_type: str,
    ) -> str:

        instructions = {

            "abstract": """
Write one compact academic abstract.

Include, when supported:
- background/problem
- objective
- methodology
- key findings
- contribution/significance

Use approximately 150–250 words.
Do not use headings inside the abstract.
""",

            "introduction": """
Write a strong academic introduction.

Structure it as connected paragraphs:

1. Research context/background
2. Importance of the problem
3. Current research direction
4. What existing studies have addressed
5. Remaining challenge/gap, only when supported
6. Motivation for further research
7. Purpose of the proposed research direction

Do not overstate novelty.
""",

            "related_work": """
Write a concise related-work section.

Synthesize the papers thematically.

Compare:
- research objectives
- methodologies
- datasets
- models
- findings
- limitations

Do not produce disconnected paper summaries.
Use transitions showing how studies relate to each other.
""",

            "methodology": """
Write a methodology section based strictly on methods supported
by the papers.

Describe, where available:
- data
- preprocessing
- model/architecture
- training
- evaluation
- experimental setup

Do not invent implementation details that are not supported.
""",

            "results_discussion": """
Write a results and discussion section.

Summarize supported findings and evaluation results.

Discuss:
- what the studies achieved
- important performance observations
- methodological differences
- implications
- limitations affecting interpretation

Never invent metrics.
""",

            "conclusion": """
Write a concise academic conclusion.

Summarize:
- central research problem
- main findings from the supplied studies
- collective contribution
- remaining challenge
- reasonable future research direction

Do not introduce new unsupported findings.
""",

            "full_paper": """
Generate a compact research-paper draft using the supplied
evidence.

Use these sections:

# Abstract
# 1. Introduction
# 2. Related Work
# 3. Methodology
# 4. Results and Discussion
# 5. Research Gap
# 6. Conclusion

Keep each section concise.

Do not fabricate citations or experimental results.
""",
        }

        return instructions.get(
            writeup_type,
            instructions[
                "introduction"
            ],
        )

    # ============================================================
    # LLM
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

        if not callable(method):

            print(
                "Paper Write-up | "
                "ChatService._call_llm unavailable."
            )

            return None

        try:

            result = method(
                prompt,
                max_tokens=1800,
            )

            if isinstance(
                result,
                str,
            ):

                result = result.strip()

                if result:
                    return result

            return None

        except TypeError:

            # Compatibility with older ChatService
            # implementations that may not accept
            # max_tokens.
            try:

                result = method(
                    prompt
                )

                if isinstance(
                    result,
                    str,
                ):

                    return result.strip()

            except Exception as exc:

                print(
                    "Paper Write-up | "
                    f"LLM fallback failed: {exc}"
                )

            return None

        except Exception as exc:

            print(
                "Paper Write-up | "
                f"LLM generation failed: {exc}"
            )

            return None

    # ============================================================
    # CLEAN OUTPUT
    # ============================================================

    def _clean_output(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = str(
            text
        ).strip()

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

        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # FALLBACK
    # ============================================================

    def _fallback_writeup(
        self,
        writeup_type: str,
        evidence: List[Dict[str, Any]],
    ) -> str:

        if not evidence:

            return (
                "No sufficient evidence was retrieved from "
                "the selected papers to generate this section."
            )

        paragraphs = []

        for item in evidence[:8]:

            text = self._extract_text(
                item
            )

            if not text:
                continue

            cleaned = (
                " ".join(
                    text.split()
                )
            )

            paragraphs.append(
                cleaned[:1000]
            )

        if not paragraphs:

            return (
                "No usable evidence was found in the selected papers."
            )

        title = self.WRITEUP_TYPES[
            writeup_type
        ]

        return (
            f"# {title}\n\n"
            + "\n\n".join(
                paragraphs
            )
        )


# ================================================================
# SINGLETON
# ================================================================

paper_writeup_service = (
    PaperWriteupService()
)