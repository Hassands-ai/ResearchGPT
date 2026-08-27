from __future__ import annotations

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

    Production-oriented academic citation and paper-analysis service.

    Responsibilities:
        1. Retrieve bibliographic evidence.
        2. Retrieve research evidence.
        3. Extract and validate metadata.
        4. Generate APA 7th and IEEE citations.
        5. Generate an academic explanation.
        6. Summarize research focus, methodology,
           findings, contribution and citation context.
        7. Keep paper-specific information evidence-grounded.

    Supports:
        - 1–10 papers
        - APA 7th
        - IEEE
        - research focus
        - methodology
        - findings
        - contribution
        - citation context
        - academic explanation
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MAX_PAPERS = 10

    RETRIEVAL_LIMIT_PER_PAPER = 5

    MAX_EVIDENCE_PER_PAPER = 25

    MAX_SOURCE_CHARS = 24000

    LLM_MAX_TOKENS = 2200

    # ============================================================
    # RETRIEVAL CATEGORIES
    # ============================================================

    CATEGORIES = [
        (
            "Bibliographic Information",
            (
                "exact paper title authors author names publication "
                "year journal conference proceedings publisher "
                "volume issue pages article number DOI doi identifier"
            ),
        ),
        (
            "Research Focus",
            (
                "research problem research question objective aim "
                "purpose motivation topic scope background study"
            ),
        ),
        (
            "Methodology",
            (
                "method methodology research design experimental "
                "setup procedure framework pipeline architecture "
                "model algorithm preprocessing training validation "
                "testing implementation"
            ),
        ),
        (
            "Dataset & Data",
            (
                "dataset data source population participants patients "
                "samples sample size images records observations "
                "annotations labels preprocessing train test "
                "validation split"
            ),
        ),
        (
            "Models & Algorithms",
            (
                "model models algorithm algorithms architecture "
                "machine learning deep learning neural network CNN "
                "Transformer classifier segmentation detection "
                "feature extraction optimization"
            ),
        ),
        (
            "Key Findings",
            (
                "results findings performance evaluation accuracy "
                "precision recall F1 AUC sensitivity specificity "
                "Dice IoU statistical results conclusion"
            ),
        ),
        (
            "Contribution",
            (
                "main contribution novelty innovation proposed "
                "approach advancement significance original "
                "contribution improvement"
            ),
        ),
        (
            "Citation Context",
            (
                "related work significance practical significance "
                "theoretical significance field contribution "
                "how this study should be cited"
            ),
        ),
        (
            "Limitations & Future Work",
            (
                "limitations weaknesses constraints challenges "
                "failure cases generalization future work "
                "recommendations unresolved problems"
            ),
        ),
    ]

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        self.api_keys = list(
            getattr(
                settings,
                "api_keys_list",
                [],
            )
            or []
        )

        self.models = list(
            getattr(
                settings,
                "models_list",
                [],
            )
            or []
        )

        self.base_url = getattr(
            settings,
            "OPENROUTER_BASE_URL",
            None,
        )

    # ============================================================
    # NORMALIZE PAPER IDS
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
    # CLEAN TEXT
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
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\x00", " ")
        )

        # Preserve paragraph boundaries.
        lines = []

        for line in text.split("\n"):

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
    # COMPACT TEXT
    # ============================================================

    def _compact_text(
        self,
        value: Any,
    ) -> str:

        return " ".join(
            self._clean_text(
                value
            ).split()
        ).strip()

    # ============================================================
    # EXTRACT RESULT LIST
    # ============================================================

    def _extract_results(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:

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
                "evidence",
                "documents",
                "chunks",
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

            if self._extract_text(
                result
            ):

                return [result]

        return []

    # ============================================================
    # EXTRACT TEXT FROM RESULT
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

                cleaned = self._clean_text(
                    value
                )

                if cleaned:
                    return cleaned

        return ""

    # ============================================================
    # EXTRACT PAPER ID
    # ============================================================

    def _extract_paper_id(
        self,
        item: Dict[str, Any],
    ) -> Optional[int]:

        keys = [
            "paper_id",
            "document_id",
            "source_paper_id",
        ]

        containers = [
            item,
            item.get("metadata") or {},
            item.get("payload") or {},
            item.get("meta") or {},
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

                try:

                    if value is not None:
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
    # EXTRACT PAPER TITLE
    # ============================================================

    def _extract_title(
        self,
        item: Dict[str, Any],
    ) -> str:

        keys = [
            "title",
            "paper_title",
            "paperTitle",
            "document_title",
            "documentTitle",
            "source_title",
            "name",
        ]

        containers = [
            item,
            item.get("metadata") or {},
            item.get("payload") or {},
            item.get("meta") or {},
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

                cleaned = self._clean_text(
                    value
                )

                if cleaned:
                    return cleaned

        return ""

    # ============================================================
    # EXTRACT RELEVANCE SCORE
    # ============================================================

    def _score(
        self,
        item: Dict[str, Any],
    ) -> float:

        for key in (
            "score",
            "similarity",
            "relevance_score",
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
    # DUPLICATE CHECK
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

        for old in existing:

            old_normalized = (
                self._compact_text(
                    old
                ).lower()
            )

            if not old_normalized:
                continue

            if normalized == old_normalized:
                return True

            if (
                len(normalized) > 80
                and len(old_normalized) > 80
            ):

                words_a = set(
                    normalized.split()
                )

                words_b = set(
                    old_normalized.split()
                )

                union = (
                    words_a | words_b
                )

                if union:

                    similarity = (
                        len(
                            words_a & words_b
                        )
                        / len(union)
                    )

                    if similarity >= 0.90:
                        return True

        return False

    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    def _retrieve_evidence(
        self,
        paper_ids: List[int],
    ) -> Dict[int, List[Dict[str, Any]]]:

        evidence = {
            paper_id: []
            for paper_id in paper_ids
        }

        for paper_id in paper_ids:

            print(
                "Citation Manager | "
                f"Processing Paper {paper_id}"
            )

            selected = []

            seen_texts = []

            # ----------------------------------------------------
            # Search all academic categories.
            # ----------------------------------------------------

            for category_name, query in (
                self.CATEGORIES
            ):

                print(
                    "Citation Manager | "
                    f"Paper {paper_id} | "
                    f"Retrieving {category_name}"
                )

                try:

                    result = (
                        multi_document_service.search(
                            query=query,
                            paper_ids=[
                                paper_id
                            ],
                            limit_per_paper=(
                                self.RETRIEVAL_LIMIT_PER_PAPER
                            ),
                        )
                    )

                except Exception as exc:

                    print(
                        "Citation Manager | "
                        f"Retrieval error | "
                        f"Paper {paper_id} | "
                        f"{category_name}: "
                        f"{exc}"
                    )

                    continue

                candidates = (
                    self._extract_results(
                        result
                    )
                )

                if not candidates:
                    continue

                # Strongest evidence first.
                candidates.sort(
                    key=self._score,
                    reverse=True,
                )

                category_count = 0

                for item in candidates:

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

                    # If metadata contains paper ID,
                    # ensure it belongs to the requested paper.
                    if (
                        item_paper_id is not None
                        and item_paper_id
                        != paper_id
                    ):
                        continue

                    text = self._extract_text(
                        item
                    )

                    if not text:
                        continue

                    if self._is_duplicate(
                        text,
                        seen_texts,
                    ):
                        continue

                    copied = dict(
                        item
                    )

                    copied["text"] = text

                    copied["category"] = (
                        category_name
                    )

                    copied["score"] = (
                        self._score(
                            item
                        )
                    )

                    title = (
                        self._extract_title(
                            item
                        )
                    )

                    if title:
                        copied[
                            "paper_title"
                        ] = title

                    selected.append(
                        copied
                    )

                    seen_texts.append(
                        text
                    )

                    category_count += 1

                    # Avoid flooding one category.
                    if category_count >= 4:
                        break

                    if (
                        len(selected)
                        >= self.MAX_EVIDENCE_PER_PAPER
                    ):
                        break

                if (
                    len(selected)
                    >= self.MAX_EVIDENCE_PER_PAPER
                ):
                    break

            # ----------------------------------------------------
            # Final strongest-evidence ordering.
            # ----------------------------------------------------

            selected.sort(
                key=self._score,
                reverse=True,
            )

            evidence[
                paper_id
            ] = selected[
                :self.MAX_EVIDENCE_PER_PAPER
            ]

            print(
                "Citation Manager | "
                f"Paper {paper_id}: "
                f"{len(evidence[paper_id])} evidence chunks"
            )

        return evidence

    # ============================================================
    # EXTRACT METADATA
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

        field_aliases = {
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
                "pageRange",
            ],
            "doi": [
                "doi",
                "DOI",
            ],
        }

        # --------------------------------------------------------
        # 1. Structured metadata first.
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

                for field, aliases in (
                    field_aliases.items()
                ):

                    if metadata[field]:
                        continue

                    for key in aliases:

                        value = container.get(
                            key
                        )

                        if value is None:
                            continue

                        cleaned = self._clean_text(
                            value
                        )

                        if cleaned:
                            metadata[field] = (
                                cleaned
                            )

                            break

        # --------------------------------------------------------
        # 2. Collect text.
        # --------------------------------------------------------

        combined_text = "\n".join(
            self._clean_text(
                item.get(
                    "text",
                    "",
                )
            )
            for item in chunks
            if isinstance(
                item,
                dict,
            )
        )

        if not combined_text:
            return metadata

        # --------------------------------------------------------
        # 3. DOI extraction.
        # --------------------------------------------------------

        if not metadata["doi"]:

            match = re.search(
                r"(?:https?://doi\.org/|doi:\s*)?"
                r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
                combined_text,
                flags=re.I,
            )

            if match:

                doi = (
                    match.group(1)
                    .rstrip(
                        ".,;:)]}"
                    )
                )

                metadata["doi"] = doi

        # --------------------------------------------------------
        # 4. Year extraction.
        # --------------------------------------------------------

        if not metadata["year"]:

            years = re.findall(
                r"\b(19\d{2}|20\d{2})\b",
                combined_text,
            )

            if years:

                # Prefer a recent publication year,
                # but stay within the paper evidence.
                metadata["year"] = (
                    years[0]
                )

        # --------------------------------------------------------
        # 5. Label-based extraction.
        # --------------------------------------------------------

        patterns = {
            "title": [
                r"(?im)^\s*title\s*[:\-]\s*(.+)$",
                r"(?im)^\s*paper\s+title\s*[:\-]\s*(.+)$",
            ],
            "authors": [
                r"(?im)^\s*authors?\s*[:\-]\s*(.+)$",
            ],
            "journal": [
                r"(?im)^\s*journal\s*[:\-]\s*(.+)$",
                r"(?im)^\s*journal\s+name\s*[:\-]\s*(.+)$",
            ],
            "conference": [
                r"(?im)^\s*conference\s*[:\-]\s*(.+)$",
                r"(?im)^\s*conference\s+name\s*[:\-]\s*(.+)$",
            ],
            "publisher": [
                r"(?im)^\s*publisher\s*[:\-]\s*(.+)$",
            ],
            "volume": [
                r"(?im)^\s*volume\s*[:\-]\s*(.+)$",
            ],
            "issue": [
                r"(?im)^\s*issue\s*[:\-]\s*(.+)$",
            ],
            "pages": [
                r"(?im)^\s*pages?\s*[:\-]\s*(.+)$",
            ],
        }

        for field, patterns_list in (
            patterns.items()
        ):

            if metadata[field]:
                continue

            for pattern in patterns_list:

                match = re.search(
                    pattern,
                    combined_text,
                )

                if not match:
                    continue

                value = self._clean_text(
                    match.group(1)
                )

                if value:

                    metadata[field] = (
                        value
                    )

                    break

        # --------------------------------------------------------
        # 6. Common DOI URL normalization.
        # --------------------------------------------------------

        if metadata["doi"]:

            metadata["doi"] = (
                metadata["doi"]
                .replace(
                    "https://doi.org/",
                    "",
                )
                .replace(
                    "http://doi.org/",
                    "",
                )
                .replace(
                    "doi:",
                    "",
                )
                .strip()
            )

        return metadata

    # ============================================================
    # BUILD SOURCE TEXT
    # ============================================================

    def _build_source_text(
        self,
        chunks: List[Dict[str, Any]],
        max_chars: int = MAX_SOURCE_CHARS,
    ) -> str:

        if not chunks:
            return ""

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
                item.get(
                    "text",
                    "",
                )
            )

            if not text:
                continue

            category = self._clean_text(
                item.get(
                    "category",
                    "",
                )
            )

            score = self._score(
                item
            )

            piece = (
                f"SOURCE {index}"
                f" | CATEGORY: "
                f"{category or 'Research Evidence'}"
                f" | RELEVANCE: {score:.4f}"
                f"\n{text}"
            )

            remaining = (
                max_chars - total
            )

            if remaining <= 250:
                break

            piece = piece[
                :remaining
            ]

            sections.append(
                piece
            )

            total += len(
                piece
            )

        return "\n\n".join(
            sections
        )

    # ============================================================
    # OPENAI CLIENT
    # ============================================================

    def _get_client(
        self,
        api_key: str,
    ) -> OpenAI:

        kwargs = {
            "api_key": api_key,
        }

        if self.base_url:
            kwargs["base_url"] = (
                self.base_url
            )

        return OpenAI(
            **kwargs
        )

    # ============================================================
    # EXTRACT JSON
    # ============================================================

    def _extract_json(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:

        if not text:
            return None

        cleaned = (
            str(text)
            .strip()
        )

        # Remove markdown fences.
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

        # Direct parse.
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

        # Find JSON object.
        start = cleaned.find(
            "{"
        )

        end = cleaned.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):

            candidate = (
                cleaned[
                    start:end + 1
                ]
            )

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

        prompt = f"""
You are PaperAxiom, an expert academic research
assistant, literature analyst and citation specialist.

Analyze the research paper identified below.

============================================================
PAPER ID
============================================================

{paper_id}

============================================================
CURRENTLY EXTRACTED METADATA
============================================================

Title:
{metadata.get("title") or "unknown"}

Authors:
{metadata.get("authors") or "unknown"}

Year:
{metadata.get("year") or "unknown"}

Journal:
{metadata.get("journal") or "unknown"}

Conference:
{metadata.get("conference") or "unknown"}

Publisher:
{metadata.get("publisher") or "unknown"}

Volume:
{metadata.get("volume") or "unknown"}

Issue:
{metadata.get("issue") or "unknown"}

Pages:
{metadata.get("pages") or "unknown"}

DOI:
{metadata.get("doi") or "unknown"}

============================================================
SOURCE EVIDENCE
============================================================

{source_text}

============================================================
CORE INSTRUCTIONS
============================================================

The source evidence is the primary factual source.

You are expected to SYNTHESIZE the paper, not simply repeat
isolated retrieved chunks.

Use your own academic language to explain what the evidence
means.

You may use general academic knowledge to clarify concepts,
but never use general knowledge to invent paper-specific facts.

Never invent:

- authors
- title
- year
- journal
- conference
- publisher
- DOI
- volume
- issue
- pages
- dataset
- sample size
- model
- algorithm
- numerical result
- experiment
- contribution
- limitation

If information is genuinely unavailable, return an empty
string for that field.

============================================================
METADATA PRIORITY
============================================================

When metadata conflicts:

1. Prefer explicit bibliographic information in the source.
2. Prefer structured document metadata when clearly associated
   with this paper.
3. Prefer repeated consistent information.
4. Never guess.

============================================================
ACADEMIC ANALYSIS
============================================================

Generate:

research_focus
    Explain what problem the paper investigates, why it matters,
    and what objective the study has.

methodology_summary
    Explain the methodology, experimental design, models,
    algorithms, preprocessing and workflow where supported.

key_findings
    Explain the major findings and preserve important reported
    numerical results.

contribution
    Explain what the study contributes to the research field.

citation_context
    Explain what type of statement another researcher could
    appropriately support by citing this paper.

academic_explanation
    Give a coherent academic explanation of the entire study.
    Do not simply list retrieved evidence.

============================================================
CITATION GENERATION
============================================================

Generate both:

APA 7th edition citation.

IEEE citation.

Do not invent missing information.

For unavailable fields, simply omit them.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly these keys:

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

============================================================
QUALITY REQUIREMENTS
============================================================

- Be academically precise.
- Be concise but informative.
- Preserve numerical findings.
- Do not fabricate information.
- Do not copy large passages.
- Do not expose reasoning.
- Do not mention retrieval.
- Do not mention chunks.
- Do not mention embeddings.
- Do not mention Qdrant.
- Do not mention prompts.
- Do not mention model selection.

Return ONLY JSON.
"""

        # --------------------------------------------------------
        # Try configured API keys and models.
        # --------------------------------------------------------

        for api_key in self.api_keys:

            if not api_key:
                continue

            for model in self.models:

                if not model:
                    continue

                try:

                    print(
                        "Citation Manager | "
                        f"Calling model: {model}"
                    )

                    client = (
                        self._get_client(
                            api_key
                        )
                    )

                    response = (
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert "
                                        "academic citation "
                                        "assistant. Return "
                                        "valid JSON only."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ],
                            temperature=0.1,
                            max_tokens=self.LLM_MAX_TOKENS,
                        )
                    )

                    if not response:
                        continue

                    if not response.choices:
                        continue

                    content = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    data = (
                        self._extract_json(
                            content
                        )
                    )

                    if data:

                        print(
                            "Citation Manager | "
                            f"Successful model: {model}"
                        )

                        return data

                except Exception as exc:

                    print(
                        "Citation Manager | "
                        f"Model failed: {model} | "
                        f"{exc}"
                    )

                    continue

        return {}

    # ============================================================
    # MERGE METADATA
    # ============================================================

    def _merge_metadata(
        self,
        metadata: Dict[str, str],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = {}

        metadata_fields = [
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

        for field in metadata_fields:

            source_value = self._clean_text(
                metadata.get(
                    field,
                    "",
                )
            )

            llm_value = self._clean_text(
                analysis.get(
                    field,
                    "",
                )
            )

            # Structured/source metadata takes priority.
            result[field] = (
                source_value
                or llm_value
            )

        analysis_fields = [
            "research_focus",
            "methodology_summary",
            "key_findings",
            "contribution",
            "citation_context",
            "academic_explanation",
        ]

        for field in analysis_fields:

            result[field] = (
                self._clean_text(
                    analysis.get(
                        field,
                        "",
                    )
                )
            )

        return result

    # ============================================================
    # APA 7 BUILDER
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

            doi_clean = (
                doi
                .replace(
                    "https://doi.org/",
                    "",
                )
                .replace(
                    "http://doi.org/",
                    "",
                )
                .replace(
                    "doi:",
                    "",
                )
                .strip()
            )

            parts.append(
                "https://doi.org/"
                + doi_clean
            )

        citation = " ".join(
            parts
        ).strip()

        return (
            citation
            or
            "Citation information unavailable "
            "from the available paper evidence."
        )

    # ============================================================
    # IEEE BUILDER
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

        parts = [
            "[1]"
        ]

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

            doi_clean = (
                doi
                .replace(
                    "https://doi.org/",
                    "",
                )
                .replace(
                    "http://doi.org/",
                    "",
                )
                .replace(
                    "doi:",
                    "",
                )
                .strip()
            )

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
            or
            "Citation information unavailable "
            "from the available paper evidence."
        )

    # ============================================================
    # FALLBACK ACADEMIC EXPLANATION
    # ============================================================

    def _fallback_explanation(
        self,
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, str]:

        # Evidence-only fallback when the LLM is unavailable.
        #
        # This deliberately does not invent conclusions.

        focus = ""

        methodology = ""

        findings = ""

        contribution = ""

        citation_context = ""

        # --------------------------------------------------------
        # Category-based evidence.
        # --------------------------------------------------------

        grouped = {}

        for item in chunks:

            category = (
                item.get(
                    "category",
                    "",
                )
            )

            text = self._clean_text(
                item.get(
                    "text",
                    "",
                )
            )

            if not text:
                continue

            grouped.setdefault(
                category,
                [],
            ).append(
                text
            )

        if grouped.get(
            "Research Focus"
        ):

            focus = " ".join(
                grouped[
                    "Research Focus"
                ][:2]
            )[:1800]

        if grouped.get(
            "Methodology"
        ):

            methodology = " ".join(
                grouped[
                    "Methodology"
                ][:3]
            )[:2200]

        if grouped.get(
            "Key Findings"
        ):

            findings = " ".join(
                grouped[
                    "Key Findings"
                ][:3]
            )[:2200]

        if grouped.get(
            "Contribution"
        ):

            contribution = " ".join(
                grouped[
                    "Contribution"
                ][:2]
            )[:1800]

        if grouped.get(
            "Citation Context"
        ):

            citation_context = " ".join(
                grouped[
                    "Citation Context"
                ][:2]
            )[:1800]

        academic_parts = []

        if focus:
            academic_parts.append(
                focus
            )

        if methodology:
            academic_parts.append(
                methodology
            )

        if findings:
            academic_parts.append(
                findings
            )

        if contribution:
            academic_parts.append(
                contribution
            )

        academic_explanation = (
            " ".join(
                academic_parts
            )
        )[:5000]

        return {
            "research_focus": focus,
            "methodology_summary": methodology,
            "key_findings": findings,
            "contribution": contribution,
            "citation_context": citation_context,
            "academic_explanation": academic_explanation,
        }

    # ============================================================
    # SOURCE COUNT / STATUS
    # ============================================================

    def _determine_status(
        self,
        metadata: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> str:

        required = [
            metadata.get(
                "title"
            ),
            metadata.get(
                "year"
            ),
        ]

        if all(
            required
        ):

            if analysis:
                return "ai_generated"

            return "evidence_grounded"

        if analysis:
            return "ai_generated_incomplete_metadata"

        return "metadata_incomplete"

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

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        if not normalized_ids:

            raise ValueError(
                "Select at least one paper."
            )

        if len(normalized_ids) > self.MAX_PAPERS:

            raise ValueError(
                "You can select a maximum "
                "of 10 papers."
            )

        print(
            "=================================================="
        )

        print(
            "Citation Manager | "
            f"Processing {len(normalized_ids)} paper(s)"
        )

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        evidence = (
            self._retrieve_evidence(
                normalized_ids
            )
        )

        citations = []

        # --------------------------------------------------------
        # Process each paper.
        # --------------------------------------------------------

        for paper_id in normalized_ids:

            print(
                "Citation Manager | "
                f"Analyzing Paper {paper_id}"
            )

            chunks = (
                evidence.get(
                    paper_id,
                    [],
                )
            )

            # ----------------------------------------------------
            # Metadata extraction
            # ----------------------------------------------------

            metadata = (
                self._extract_metadata_from_chunks(
                    chunks
                )
            )

            # ----------------------------------------------------
            # Source context
            # ----------------------------------------------------

            source_text = (
                self._build_source_text(
                    chunks,
                    max_chars=self.MAX_SOURCE_CHARS,
                )
            )

            # ----------------------------------------------------
            # LLM analysis
            # ----------------------------------------------------

            analysis = (
                self._generate_llm_analysis(
                    paper_id=paper_id,
                    metadata=metadata,
                    source_text=(
                        source_text
                        or
                        "No sufficient paper evidence "
                        "was retrieved."
                    ),
                )
            )

            # ----------------------------------------------------
            # If LLM fails, create evidence fallback.
            # ----------------------------------------------------

            if not analysis:

                analysis = (
                    self._fallback_explanation(
                        metadata=metadata,
                        chunks=chunks,
                    )
                )

            # ----------------------------------------------------
            # Merge.
            # ----------------------------------------------------

            merged = (
                self._merge_metadata(
                    metadata,
                    analysis,
                )
            )

            # ----------------------------------------------------
            # Build citations locally.
            # ----------------------------------------------------

            apa = self._build_apa(
                merged
            )

            ieee = self._build_ieee(
                merged
            )

            # ----------------------------------------------------
            # Human-readable title fallback.
            # ----------------------------------------------------

            title = (
                merged.get(
                    "title"
                )
                or
                "Title not available in the "
                "retrieved evidence."
            )

            # ----------------------------------------------------
            # Status.
            # ----------------------------------------------------

            citation_status = (
                self._determine_status(
                    metadata=merged,
                    analysis=analysis,
                )
            )

            # ----------------------------------------------------
            # Final citation object.
            # ----------------------------------------------------

            citations.append(
                {
                    "paper_id": paper_id,

                    "title": title,

                    "authors": merged.get(
                        "authors",
                        "",
                    ),

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
                        or
                        "Research focus could not be "
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

                    "citation_status": (
                        citation_status
                    ),
                }
            )

        # --------------------------------------------------------
        # Final result.
        # --------------------------------------------------------

        total_sources = sum(
            len(items)
            for items in evidence.values()
        )

        result = {
            "papers_count": len(
                normalized_ids
            ),

            "paper_ids": normalized_ids,

            "source_count": total_sources,

            "citations": citations,

            "generation_status": (
                "ai_generated"
            ),
        }

        print(
            "Citation Manager | "
            f"Completed | "
            f"papers={len(normalized_ids)} | "
            f"sources={total_sources}"
        )

        print(
            "=================================================="
        )

        return result


# ================================================================
# SERVICE INSTANCE
# IMPORTANT:
# routes.py imports this exact object.
# ================================================================

citation_manager_service = (
    CitationManagerService()
)