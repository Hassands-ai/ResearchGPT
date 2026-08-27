from typing import Any, Dict, List, Optional
import json
import re

from openai import OpenAI

from app.core.config import settings
from app.services.multi_document_service import (
    multi_document_service,
)


class CitationManagerService:
    """
    PaperAxiom Citation Manager.

    Responsibilities:
        1. Retrieve bibliographic and research evidence.
        2. Extract citation metadata from the paper evidence.
        3. Generate APA 7th and IEEE citations.
        4. Explain the paper in concise academic language.
        5. Keep all paper-specific information evidence-grounded.

    Supports:
        - 1–10 papers
        - APA 7th
        - IEEE
        - research focus
        - contribution
        - methodology
        - key findings
        - citation context
        - concise academic explanation
    """

    CATEGORIES = [
        (
            "Bibliographic Information",
            (
                "paper title authors publication year journal "
                "conference DOI publisher volume issue pages "
                "article number citation reference"
            ),
        ),
        (
            "Research Focus",
            (
                "research problem research question objective "
                "research aim motivation topic scope"
            ),
        ),
        (
            "Methodology",
            (
                "method methodology architecture model algorithm "
                "experimental setup research design procedure"
            ),
        ),
        (
            "Key Findings",
            (
                "results findings performance evaluation metrics "
                "accuracy precision recall F1 AUC conclusions"
            ),
        ),
        (
            "Contribution",
            (
                "contribution novelty significance proposed approach "
                "innovation advancement"
            ),
        ),
        (
            "Citation Context",
            (
                "how this study contributes to the field "
                "related work importance practical significance"
            ),
        ),
        (
            "Limitations",
            (
                "limitations weaknesses challenges constraints "
                "failure cases unresolved issues"
            ),
        ),
    ]

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        self.api_keys = settings.api_keys_list
        self.models = settings.models_list
        self.base_url = settings.OPENROUTER_BASE_URL

    # ============================================================
    # NORMALIZE PAPER IDS
    # ============================================================

    def _normalize_ids(
        self,
        paper_ids: List[int],
    ) -> List[int]:

        result = []

        for value in paper_ids or []:

            try:
                paper_id = int(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if paper_id not in result:
                result.append(paper_id)

        return result

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    def _clean_text(
        self,
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(value)

        text = (
            text
            .replace("\r", " ")
            .replace("\n", " ")
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
    ) -> Dict[int, List[Dict[str, Any]]]:

        evidence = {}

        for paper_id in paper_ids:

            chunks = []

            for category_name, query in self.CATEGORIES:

                print(
                    "Citation Manager | "
                    f"Retrieving: {query}"
                )

                try:

                    result = (
                        multi_document_service.search(
                            query=query,
                            paper_ids=[paper_id],
                            limit_per_paper=2,
                        )
                    )

                except Exception as exc:

                    print(
                        "Citation Manager | "
                        f"Retrieval error for "
                        f"Paper {paper_id}: {exc}"
                    )

                    continue

                candidates = []

                if isinstance(
                    result,
                    dict,
                ):

                    candidates = (
                        result.get("results")
                        or result.get("papers")
                        or result.get("evidence")
                        or []
                    )

                elif isinstance(
                    result,
                    list,
                ):

                    candidates = result

                if not isinstance(
                    candidates,
                    list,
                ):
                    continue

                for item in candidates:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    text = self._clean_text(
                        item.get("text")
                        or item.get("content")
                        or item.get("chunk")
                        or ""
                    )

                    if not text:
                        continue

                    enriched = dict(item)

                    enriched["text"] = text
                    enriched["category"] = (
                        category_name
                    )

                    chunks.append(
                        enriched
                    )

            evidence[paper_id] = (
                self._deduplicate(chunks)
            )

            print(
                "Citation Manager evidence | "
                f"Paper {paper_id}: "
                f"{len(evidence[paper_id])} chunks"
            )

        return evidence

    # ============================================================
    # DEDUPLICATE
    # ============================================================

    def _deduplicate(
        self,
        chunks: List[Any],
    ) -> List[Dict[str, Any]]:

        output = []
        seen = set()

        for item in chunks:

            if not isinstance(
                item,
                dict,
            ):
                continue

            text = self._clean_text(
                item.get("text")
                or item.get("content")
                or item.get("chunk")
                or ""
            )

            if not text:
                continue

            key = text[:700].lower()

            if key in seen:
                continue

            seen.add(key)

            copied = dict(item)
            copied["text"] = text

            output.append(
                copied
            )

        return output

    # ============================================================
    # EXTRACT METADATA FROM RETRIEVED CHUNKS
    # ============================================================

    def _extract_metadata_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, str]:

        metadata = {
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "conference": "",
            "publisher": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": "",
        }

        possible_fields = {
            "title": [
                "title",
                "paper_title",
                "paperTitle",
                "document_title",
                "documentTitle",
                "source_title",
            ],
            "authors": [
                "authors",
                "author",
                "paper_authors",
                "paperAuthors",
                "author_names",
            ],
            "year": [
                "year",
                "publication_year",
                "publicationYear",
                "published_year",
            ],
            "journal": [
                "journal",
                "journal_name",
                "journalName",
            ],
            "conference": [
                "conference",
                "conference_name",
                "conferenceName",
            ],
            "publisher": [
                "publisher",
            ],
            "volume": [
                "volume",
            ],
            "issue": [
                "issue",
                "number",
            ],
            "pages": [
                "pages",
                "page",
                "page_range",
            ],
            "doi": [
                "doi",
                "DOI",
            ],
        }

        # --------------------------------------------------------
        # First inspect structured metadata
        # --------------------------------------------------------

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            containers = [
                chunk,
                chunk.get("metadata") or {},
                chunk.get("payload") or {},
                chunk.get("meta") or {},
            ]

            for container in containers:

                if not isinstance(
                    container,
                    dict,
                ):
                    continue

                for field, keys in possible_fields.items():

                    if metadata[field]:
                        continue

                    for key in keys:

                        value = container.get(
                            key
                        )

                        if value is None:
                            continue

                        value = self._clean_text(
                            value
                        )

                        if value:
                            metadata[field] = value
                            break

        # --------------------------------------------------------
        # Then inspect text itself
        # --------------------------------------------------------

        combined_text = "\n".join(
            self._clean_text(
                item.get("text", "")
            )
            for item in chunks
            if isinstance(item, dict)
        )

        if not combined_text:
            return metadata

        # DOI
        if not metadata["doi"]:

            match = re.search(
                r"(?:https?://doi\.org/|doi:\s*)"
                r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
                combined_text,
                flags=re.I,
            )

            if match:
                metadata["doi"] = (
                    match.group(1)
                    .rstrip(".,;)")
                )

        # Year
        if not metadata["year"]:

            matches = re.findall(
                r"\b(19\d{2}|20\d{2})\b",
                combined_text,
            )

            if matches:
                metadata["year"] = matches[0]

        # --------------------------------------------------------
        # Look for common bibliographic labels
        # --------------------------------------------------------

        patterns = {
            "title": [
                r"(?im)^title\s*[:\-]\s*(.+)$",
                r"(?im)^paper title\s*[:\-]\s*(.+)$",
            ],
            "journal": [
                r"(?im)^journal\s*[:\-]\s*(.+)$",
                r"(?im)^journal name\s*[:\-]\s*(.+)$",
            ],
            "conference": [
                r"(?im)^conference\s*[:\-]\s*(.+)$",
                r"(?im)^conference name\s*[:\-]\s*(.+)$",
            ],
            "publisher": [
                r"(?im)^publisher\s*[:\-]\s*(.+)$",
            ],
            "volume": [
                r"(?im)^volume\s*[:\-]\s*(.+)$",
            ],
            "issue": [
                r"(?im)^issue\s*[:\-]\s*(.+)$",
            ],
            "pages": [
                r"(?im)^pages?\s*[:\-]\s*(.+)$",
            ],
            "authors": [
                r"(?im)^authors?\s*[:\-]\s*(.+)$",
            ],
        }

        for field, field_patterns in patterns.items():

            if metadata[field]:
                continue

            for pattern in field_patterns:

                match = re.search(
                    pattern,
                    combined_text,
                )

                if match:

                    value = self._clean_text(
                        match.group(1)
                    )

                    if value:
                        metadata[field] = value
                        break

        return metadata

    # ============================================================
    # BUILD SOURCE TEXT FOR LLM
    # ============================================================

    def _build_source_text(
        self,
        chunks: List[Dict[str, Any]],
        max_chars: int = 12000,
    ) -> str:

        sections = []

        total = 0

        for index, item in enumerate(
            chunks,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            text = self._clean_text(
                item.get("text", "")
            )

            if not text:
                continue

            category = self._clean_text(
                item.get(
                    "category",
                    "",
                )
            )

            piece = (
                f"SOURCE {index}"
                + (
                    f" | {category}"
                    if category
                    else ""
                )
                + f"\n{text}"
            )

            remaining = (
                max_chars - total
            )

            if remaining <= 200:
                break

            piece = piece[:remaining]

            sections.append(
                piece
            )

            total += len(piece)

        return "\n\n".join(
            sections
        )

    # ============================================================
    # LLM CLIENT
    # ============================================================

    def _get_client(
        self,
        api_key: str,
    ) -> OpenAI:

        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    # ============================================================
    # JSON EXTRACTION
    # ============================================================

    def _extract_json(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:

        if not text:
            return None

        cleaned = text.strip()

        # Remove markdown code fences
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.I,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:

            data = json.loads(
                cleaned
            )

            if isinstance(
                data,
                dict,
            ):
                return data

        except Exception:
            pass

        # Try to locate JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:

            candidate = cleaned[
                start:end + 1
            ]

            try:

                data = json.loads(
                    candidate
                )

                if isinstance(
                    data,
                    dict,
                ):
                    return data

            except Exception:
                pass

        return None

    # ============================================================
    # LLM ANALYSIS
    # ============================================================

    def _generate_llm_analysis(
        self,
        paper_id: int,
        metadata: Dict[str, str],
        source_text: str,
    ) -> Dict[str, Any]:

        if not self.api_keys:

            print(
                "Citation Manager | "
                "No OpenRouter API keys configured."
            )

            return {}

        if not self.models:

            print(
                "Citation Manager | "
                "No OpenRouter models configured."
            )

            return {}

        title_hint = (
            metadata.get("title")
            or "unknown title"
        )

        prompt = f"""
You are PaperAxiom, an expert academic research
assistant and citation specialist.

Analyze ONE research paper using the supplied evidence.

PAPER ID:
{paper_id}

KNOWN METADATA:
Title: {metadata.get("title") or "unknown"}
Authors: {metadata.get("authors") or "unknown"}
Year: {metadata.get("year") or "unknown"}
Journal: {metadata.get("journal") or "unknown"}
Conference: {metadata.get("conference") or "unknown"}
Publisher: {metadata.get("publisher") or "unknown"}
Volume: {metadata.get("volume") or "unknown"}
Issue: {metadata.get("issue") or "unknown"}
Pages: {metadata.get("pages") or "unknown"}
DOI: {metadata.get("doi") or "unknown"}

IMPORTANT:

The supplied document evidence is the PRIMARY source.

Your task is NOT to invent missing bibliographic
information.

Extract information only when it is supported by the
document evidence.

If a field is genuinely unavailable, return an empty
string for that field.

Do not write "Not available" inside the citation fields.

Never invent authors, title, year, journal, conference,
DOI, volume, issue, or page numbers.

If the document provides the information in a different
format, normalize it carefully.

==================================================
PAPER EVIDENCE
==================================================

{source_text}

==================================================
TASK
==================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "title": "",
  "authors": "",
  "year": "",
  "journal": "",
  "conference": "",
  "publisher": "",
  "volume": "",
  "issue": "",
  "pages": "",
  "doi": "",
  "apa_7": "",
  "ieee": "",
  "research_focus": "",
  "methodology_summary": "",
  "key_findings": "",
  "contribution": "",
  "citation_context": "",
  "academic_explanation": ""
}}

==================================================
CITATION RULES
==================================================

APA 7:

Use the standard scholarly structure.

Author(s). (Year). Title. Journal/Conference,
volume(issue), pages. DOI

Do not fabricate missing fields.

IEEE:

[1] Author(s), "Title," Journal/Conference,
vol., no., pp., year, doi.

Again, do not invent missing information.

If authors are unavailable, create the citation using
only information supported by the evidence.

Do NOT write:

"Partial citation — authors not available."

Instead simply omit the unavailable author field.

==================================================
RESEARCH FOCUS
==================================================

Explain in 1–2 concise sentences what the paper
investigates and why.

==================================================
METHODOLOGY SUMMARY
==================================================

Explain the main methodology in 1–3 sentences.

==================================================
KEY FINDINGS
==================================================

Summarize the most important findings in 1–3 sentences.

Preserve reported numerical results when available.

==================================================
CONTRIBUTION
==================================================

Explain what the study contributes to the field.

==================================================
CITATION CONTEXT
==================================================

Explain when another researcher would appropriately
cite this paper.

==================================================
ACADEMIC EXPLANATION
==================================================

Write a concise, clear paragraph explaining the paper
in your own words.

Do not copy large portions of the source.

Do not expose reasoning.

Do not mention retrieval, chunks, embeddings, Qdrant,
prompts, or model selection.

Return ONLY JSON.
"""

        for api_key in self.api_keys:

            for model in self.models:

                try:

                    print(
                        "Citation Manager | "
                        f"Generating with model: {model}"
                    )

                    client = self._get_client(
                        api_key
                    )

                    response = (
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a precise "
                                        "academic citation "
                                        "assistant. "
                                        "Return valid JSON "
                                        "only."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ],
                            temperature=0.1,
                            max_tokens=1200,
                        )
                    )

                    if not response:
                        continue

                    if not response.choices:
                        continue

                    content = (
                        response.choices[0]
                        .message
                        .content
                    )

                    data = self._extract_json(
                        content
                    )

                    if data:

                        print(
                            "Citation Manager | "
                            f"Success with model: {model}"
                        )

                        return data

                except Exception as exc:

                    print(
                        "Citation Manager | "
                        f"Failed → model: {model} | "
                        f"error: {exc}"
                    )

                    continue

        return {}

    # ============================================================
    # MERGE METADATA + LLM
    # ============================================================

    def _merge_metadata(
        self,
        metadata: Dict[str, str],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = {}

        fields = [
            "title",
            "authors",
            "year",
            "journal",
            "conference",
            "publisher",
            "volume",
            "issue",
            "pages",
            "doi",
        ]

        for field in fields:

            llm_value = self._clean_text(
                analysis.get(
                    field,
                    "",
                )
            )

            metadata_value = self._clean_text(
                metadata.get(
                    field,
                    "",
                )
            )

            result[field] = (
                llm_value
                or metadata_value
            )

        # LLM-generated explanatory fields
        result["research_focus"] = (
            self._clean_text(
                analysis.get(
                    "research_focus",
                    "",
                )
            )
        )

        result["methodology_summary"] = (
            self._clean_text(
                analysis.get(
                    "methodology_summary",
                    "",
                )
            )
        )

        result["key_findings"] = (
            self._clean_text(
                analysis.get(
                    "key_findings",
                    "",
                )
            )
        )

        result["contribution"] = (
            self._clean_text(
                analysis.get(
                    "contribution",
                    "",
                )
            )
        )

        result["citation_context"] = (
            self._clean_text(
                analysis.get(
                    "citation_context",
                    "",
                )
            )
        )

        result["academic_explanation"] = (
            self._clean_text(
                analysis.get(
                    "academic_explanation",
                    "",
                )
            )
        )

        return result

    # ============================================================
    # BUILD CITATION FALLBACKS
    # ============================================================

    def _build_apa(
        self,
        data: Dict[str, Any],
    ) -> str:

        authors = self._clean_text(
            data.get(
                "authors",
                "",
            )
        )

        year = self._clean_text(
            data.get(
                "year",
                "",
            )
        )

        title = self._clean_text(
            data.get(
                "title",
                "",
            )
        )

        journal = self._clean_text(
            data.get(
                "journal",
                "",
            )
        )

        conference = self._clean_text(
            data.get(
                "conference",
                "",
            )
        )

        volume = self._clean_text(
            data.get(
                "volume",
                "",
            )
        )

        issue = self._clean_text(
            data.get(
                "issue",
                "",
            )
        )

        pages = self._clean_text(
            data.get(
                "pages",
                "",
            )
        )

        doi = self._clean_text(
            data.get(
                "doi",
                "",
            )
        )

        container = (
            journal
            or conference
        )

        parts = []

        if authors:
            parts.append(
                f"{authors}."
            )

        if year:
            parts.append(
                f"({year})."
            )

        if title:
            parts.append(
                f"{title}."
            )

        if container:

            container_part = (
                f"*{container}*"
            )

            if volume:
                container_part += (
                    f", *{volume}*"
                )

            if issue:
                container_part += (
                    f"({issue})"
                )

            if pages:
                container_part += (
                    f", {pages}"
                )

            container_part += "."

            parts.append(
                container_part
            )

        if doi:

            doi_clean = doi

            if not doi_clean.lower().startswith(
                "http"
            ):

                doi_clean = (
                    "https://doi.org/"
                    + doi_clean
                )

            parts.append(
                doi_clean
            )

        citation = " ".join(
            parts
        ).strip()

        if citation:
            return citation

        return (
            title
            or "Citation information unavailable "
            "from the document."
        )

    # ============================================================
    # IEEE
    # ============================================================

    def _build_ieee(
        self,
        data: Dict[str, Any],
    ) -> str:

        authors = self._clean_text(
            data.get(
                "authors",
                "",
            )
        )

        title = self._clean_text(
            data.get(
                "title",
                "",
            )
        )

        year = self._clean_text(
            data.get(
                "year",
                "",
            )
        )

        journal = self._clean_text(
            data.get(
                "journal",
                "",
            )
        )

        conference = self._clean_text(
            data.get(
                "conference",
                "",
            )
        )

        volume = self._clean_text(
            data.get(
                "volume",
                "",
            )
        )

        issue = self._clean_text(
            data.get(
                "issue",
                "",
            )
        )

        pages = self._clean_text(
            data.get(
                "pages",
                "",
            )
        )

        doi = self._clean_text(
            data.get(
                "doi",
                "",
            )
        )

        container = (
            journal
            or conference
        )

        parts = ["[1]"]

        if authors:
            parts.append(
                f"{authors},"
            )

        if title:
            parts.append(
                f'"{title},"'
            )

        if container:
            parts.append(
                f"*{container}*,"
            )

        if volume:
            parts.append(
                f"vol. {volume},"
            )

        if issue:
            parts.append(
                f"no. {issue},"
            )

        if pages:
            parts.append(
                f"pp. {pages},"
            )

        if year:
            parts.append(
                f"{year},"
            )

        if doi:

            doi_clean = doi

            if doi_clean.lower().startswith(
                "https://doi.org/"
            ):

                doi_clean = doi_clean[
                    len(
                        "https://doi.org/"
                    ):
                ]

            parts.append(
                f"doi: {doi_clean}"
            )

        citation = " ".join(
            parts
        ).strip()

        if citation != "[1]":
            return citation

        return (
            title
            or "Citation information unavailable "
            "from the document."
        )

    # ============================================================
    # PUBLIC GENERATE METHOD
    # ============================================================

    def generate(
        self,
        paper_ids: List[int],
    ) -> Dict[str, Any]:

        normalized_ids = (
            self._normalize_ids(
                paper_ids
            )
        )

        if not normalized_ids:

            raise ValueError(
                "Select at least one paper."
            )

        if len(normalized_ids) > 10:

            raise ValueError(
                "You can select a maximum "
                "of 10 papers."
            )

        # --------------------------------------------------------
        # RETRIEVE
        # --------------------------------------------------------

        evidence = (
            self._retrieve_evidence(
                normalized_ids
            )
        )

        citations = []

        # --------------------------------------------------------
        # PROCESS EACH PAPER
        # --------------------------------------------------------

        for paper_id in normalized_ids:

            chunks = evidence.get(
                paper_id,
                [],
            )

            metadata = (
                self._extract_metadata_from_chunks(
                    chunks
                )
            )

            source_text = (
                self._build_source_text(
                    chunks,
                    max_chars=12000,
                )
            )

            if not source_text:

                source_text = (
                    "No sufficient evidence was "
                    "retrieved from this paper."
                )

            analysis = (
                self._generate_llm_analysis(
                    paper_id=paper_id,
                    metadata=metadata,
                    source_text=source_text,
                )
            )

            merged = self._merge_metadata(
                metadata,
                analysis,
            )

            # ----------------------------------------------------
            # Build citations locally when possible.
            # This prevents the LLM from producing malformed
            # citation strings.
            # ----------------------------------------------------

            apa = self._build_apa(
                merged
            )

            ieee = self._build_ieee(
                merged
            )

            title = (
                merged.get(
                    "title"
                )
                or "Title not available in "
                "the retrieved evidence."
            )

            authors = (
                merged.get(
                    "authors"
                )
                or ""
            )

            # ----------------------------------------------------
            # Status
            # ----------------------------------------------------

            required_metadata = [
                merged.get("title"),
                merged.get("year"),
            ]

            if all(
                required_metadata
            ):

                citation_status = (
                    "evidence_grounded"
                )

            else:

                citation_status = (
                    "metadata_incomplete"
                )

            # ----------------------------------------------------
            # Add final item
            # ----------------------------------------------------

            citations.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "year": merged.get(
                        "year",
                        "",
                    ),
                    "journal": merged.get(
                        "journal",
                        "",
                    ),
                    "conference": merged.get(
                        "conference",
                        "",
                    ),
                    "publisher": merged.get(
                        "publisher",
                        "",
                    ),
                    "volume": merged.get(
                        "volume",
                        "",
                    ),
                    "issue": merged.get(
                        "issue",
                        "",
                    ),
                    "pages": merged.get(
                        "pages",
                        "",
                    ),
                    "doi": merged.get(
                        "doi",
                        "",
                    ),
                    "apa_7": apa,
                    "ieee": ieee,
                    "research_focus": (
                        merged.get(
                            "research_focus",
                            "",
                        )
                        or "Research focus could not be "
                        "determined from the available "
                        "paper evidence."
                    ),
                    "methodology_summary": (
                        merged.get(
                            "methodology_summary",
                            "",
                        )
                    ),
                    "key_findings": (
                        merged.get(
                            "key_findings",
                            "",
                        )
                    ),
                    "contribution": (
                        merged.get(
                            "contribution",
                            "",
                        )
                    ),
                    "citation_context": (
                        merged.get(
                            "citation_context",
                            "",
                        )
                    ),
                    "academic_explanation": (
                        merged.get(
                            "academic_explanation",
                            "",
                        )
                    ),
                    "evidence": chunks,
                    "source_text": source_text,
                    "citation_status": citation_status,
                }
            )

        # --------------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------------

        return {
            "papers_count": len(
                normalized_ids
            ),
            "source_count": sum(
                len(items)
                for items in evidence.values()
            ),
            "citations": citations,
            "generation_status": (
                "ai_generated"
            ),
        }


# ============================================================
# SERVICE INSTANCE
# IMPORTANT: routes.py imports this object.
# ============================================================

citation_manager_service = (
    CitationManagerService()
)