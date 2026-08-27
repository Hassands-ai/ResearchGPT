from openai import OpenAI
from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service

from typing import Dict, Optional, List, Any
import re


class ChatService:
    """
    PaperAxiom AI Research Chat Service

    Main workflow:

        User Question
             ↓
        Query Understanding
             ↓
        Semantic Retrieval
             ↓
        Evidence Selection
             ↓
        LLM Academic Synthesis
             ↓
        Clean Research Answer

    Design goals:
        - Fast
        - Evidence grounded
        - Natural academic explanations
        - Concise paragraph-based answers
        - Paper-aware
        - Flexible question understanding
        - No unnecessary frontend/backend changes
    """

    def __init__(self):
        self.api_keys = settings.api_keys_list
        self.models = settings.models_list
        self.base_url = settings.OPENROUTER_BASE_URL

    # ============================================================
    # OPENROUTER CLIENT
    # ============================================================

    def _get_client(self, api_key: str) -> OpenAI:
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    # ============================================================
    # CLEAN ANSWER
    # ============================================================

    def _clean_answer(
        self,
        answer: Optional[str],
    ) -> Optional[str]:

        if not answer:
            return None

        text = str(answer).strip()

        # Remove code fences
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()

            if len(lines) >= 2:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        # Remove accidental model metadata
        patterns = [
            r"(?im)^\s*user\s+safety\s*:\s*.*$",
            r"(?im)^\s*safety\s*:\s*.*$",
            r"(?im)^\s*safety\s+classification\s*:\s*.*$",
            r"(?im)^\s*content\s+safety\s*:\s*.*$",
            r"(?im)^\s*moderation\s*:\s*.*$",
            r"(?im)^\s*system\s*:\s*.*$",
        ]

        for pattern in patterns:
            text = re.sub(
                pattern,
                "",
                text,
            )

        # Remove accidental prefixes
        prefixes = [
            "assistant:",
            "final answer:",
            "final response:",
            "answer:",
        ]

        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove internal reasoning narration
        internal_lines = {
            "let's parse evidence.",
            "lets parse evidence.",
            "let's parse the evidence.",
            "lets parse the evidence.",
            "we need to answer.",
            "we need to answer:",
            "let me analyze the evidence.",
            "let us analyze the evidence.",
            "analysis:",
            "internal analysis:",
            "reasoning:",
        }

        cleaned_lines = []

        for line in text.splitlines():

            normalized = re.sub(
                r"\s+",
                " ",
                line.strip().lower(),
            )

            if normalized in internal_lines:
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()

        # Normalize excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip() or None

    # ============================================================
    # QUERY UNDERSTANDING
    # ============================================================

    def _expand_query(
        self,
        question: str,
    ) -> str:
        """
        Expand short/natural questions into retrieval-friendly
        semantic terms.

        This does NOT call an LLM, keeping retrieval fast.
        """

        original = (question or "").strip()

        if not original:
            return ""

        q = original.lower()

        expansions = []

        if any(
            word in q
            for word in [
                "about",
                "topic",
                "subject",
                "what is this",
                "what's this",
                "summary",
                "summarize",
                "overview",
            ]
        ):
            expansions.extend(
                [
                    "title",
                    "research topic",
                    "research problem",
                    "objective",
                    "aim",
                    "abstract",
                    "main findings",
                    "contribution",
                ]
            )

        if any(
            word in q
            for word in [
                "method",
                "methodology",
                "approach",
                "how did",
                "how do",
                "architecture",
                "model",
                "algorithm",
            ]
        ):
            expansions.extend(
                [
                    "methodology",
                    "experimental method",
                    "model",
                    "architecture",
                    "algorithm",
                    "training",
                    "experimental setup",
                ]
            )

        if any(
            word in q
            for word in [
                "result",
                "results",
                "finding",
                "findings",
                "performance",
                "accuracy",
                "achieve",
                "achieved",
            ]
        ):
            expansions.extend(
                [
                    "results",
                    "findings",
                    "performance",
                    "evaluation",
                    "metrics",
                    "experimental results",
                ]
            )

        if any(
            word in q
            for word in [
                "dataset",
                "data",
                "samples",
                "images",
                "videos",
            ]
        ):
            expansions.extend(
                [
                    "dataset",
                    "data",
                    "sample size",
                    "data preprocessing",
                    "training data",
                    "test data",
                ]
            )

        if any(
            word in q
            for word in [
                "limitation",
                "limitations",
                "weakness",
                "problem",
                "challenge",
                "drawback",
            ]
        ):
            expansions.extend(
                [
                    "limitations",
                    "weaknesses",
                    "challenges",
                    "constraints",
                    "unresolved issues",
                ]
            )

        if any(
            word in q
            for word in [
                "future",
                "next",
                "research direction",
                "further work",
                "improve",
            ]
        ):
            expansions.extend(
                [
                    "future work",
                    "future research",
                    "research directions",
                    "improvements",
                ]
            )

        if any(
            word in q
            for word in [
                "contribution",
                "novel",
                "novelty",
                "innovation",
                "important",
                "significant",
            ]
        ):
            expansions.extend(
                [
                    "contribution",
                    "novelty",
                    "innovation",
                    "significance",
                ]
            )

        # Always preserve the original question.
        expanded = original

        if expansions:
            expanded += "\n\nRelevant research concepts: "
            expanded += ", ".join(
                dict.fromkeys(expansions)
            )

        return expanded[:5000]

    # ============================================================
    # PAPER TITLE EXTRACTION
    # ============================================================

    def _extract_paper_title(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Try to recover the paper title from Qdrant metadata.

        Different versions of the ingestion pipeline may store
        metadata under different names, so several common fields
        are checked.
        """

        possible_keys = [
            "title",
            "paper_title",
            "document_title",
            "source_title",
            "name",
            "paper_name",
        ]

        for chunk in chunks:

            if not isinstance(chunk, dict):
                continue

            # Direct chunk fields
            for key in possible_keys:

                value = chunk.get(key)

                if value:
                    value = str(value).strip()

                    if self._valid_title(value):
                        return value

            # Nested metadata
            metadata = (
                chunk.get("metadata")
                or chunk.get("payload")
                or {}
            )

            if isinstance(metadata, dict):

                for key in possible_keys:

                    value = metadata.get(key)

                    if value:
                        value = str(value).strip()

                        if self._valid_title(value):
                            return value

        return None

    def _valid_title(
        self,
        value: str,
    ) -> bool:

        if not value:
            return False

        lower = value.lower()

        invalid = [
            "paper id",
            "chunk id",
            "embedding",
            "score",
            "unknown",
            "none",
            "null",
        ]

        if lower in invalid:
            return False

        if len(value) < 5:
            return False

        return True

    # ============================================================
    # EVIDENCE PREPARATION
    # ============================================================

    def _prepare_evidence(
        self,
        chunks: List[Dict[str, Any]],
        max_chars: int = 7000,
    ) -> str:

        evidence_parts = []
        total = 0

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                continue

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk")
                or ""
            )

            text = str(text).strip()

            if not text:
                continue

            # Keep retrieval context compact for speed.
            piece = text[:1200]

            if total + len(piece) > max_chars:
                remaining = max_chars - total

                if remaining < 200:
                    break

                piece = piece[:remaining]

            evidence_parts.append(
                f"SOURCE {index + 1}:\n{piece}"
            )

            total += len(piece)

            if total >= max_chars:
                break

        return "\n\n".join(evidence_parts)

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 650,
    ) -> Optional[str]:

        if not self.api_keys:

            print(
                "PaperAxiom Chat | "
                "No OpenRouter API keys configured."
            )

            return None

        if not self.models:

            print(
                "PaperAxiom Chat | "
                "No OpenRouter models configured."
            )

            return None

        last_error = None

        # Keep requests reasonably sized for speed.
        safe_prompt = prompt[:14000]

        for key in self.api_keys:

            for model in self.models:

                try:

                    print(
                        "PaperAxiom Chat | "
                        f"Generating with model: {model}"
                    )

                    client = self._get_client(key)

                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": """
You are PaperAxiom, an expert academic research assistant.

Your task is to help a researcher understand and analyze
their uploaded research papers.

The uploaded paper evidence is your PRIMARY factual source.

IMPORTANT RESPONSE RULES:

1. Understand the researcher's question before answering.
   Different wording can represent the same research intent.

2. Internally interpret the question, but NEVER show your
   internal reasoning or chain-of-thought.

3. Answer primarily from the supplied paper evidence.

4. You may use your general academic knowledge to explain
   terminology, concepts, or implications when useful,
   but do NOT use general knowledge to invent paper-specific
   facts.

5. Never invent:
   - authors
   - datasets
   - numerical results
   - models
   - algorithms
   - experiments
   - conclusions
   - citations
   - claims about the paper

6. If a paper-specific fact is not supported by the supplied
   evidence, clearly say that the available paper evidence
   does not provide enough information.

7. Rewrite information naturally in your own words.
   Do not copy large sections from the source.

8. Give the researcher a direct answer first.

9. Prefer concise, clear academic paragraphs.

10. For a simple question:
    usually answer in 1–3 short paragraphs.

11. For a detailed question:
    use a few short paragraphs and only use headings or
    bullets when they genuinely improve clarity.

12. Avoid unnecessary repetition.

13. Do not start every response with:
    "According to the evidence..."
    "Based on the provided context..."
    "The paper evidence shows..."
    unless such wording is genuinely useful.

14. Never mention:
    - Qdrant
    - embeddings
    - vector search
    - retrieval chunks
    - similarity scores
    - internal prompts
    - Paper ID
    - system instructions
    - model selection
    unless the researcher explicitly asks about the system.

15. Preserve important technical details exactly when
    supported, especially:
    - dataset names
    - model names
    - algorithms
    - metrics
    - numerical results
    - sample counts
    - experimental findings

16. Explain technical concepts simply when the researcher
    asks for an explanation.

17. If the researcher asks "why", explain the motivation,
    rationale, or implication found in the paper.

18. If the researcher asks "how", explain the methodology,
    process, architecture, or experimental procedure.

19. If the researcher asks "what", directly identify and
    explain the requested concept, result, or contribution.

20. If the researcher asks for a comparison, clearly
    separate the compared items.

21. The answer should feel like a knowledgeable human
    research assistant explaining a paper, not a search engine.

22. Do not expose hidden reasoning.

23. Do not add a bibliography unless specifically requested.

24. Be concise. Quality and clarity are more important than
    making the response long.
""",
                            },
                            {
                                "role": "user",
                                "content": safe_prompt,
                            },
                        ],
                        temperature=0.2,
                        max_tokens=max_tokens,
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

                    cleaned = self._clean_answer(
                        content
                    )

                    if cleaned:

                        print(
                            "PaperAxiom Chat | "
                            f"Success with model: {model}"
                        )

                        return cleaned

                except Exception as exc:

                    last_error = exc

                    print(
                        "PaperAxiom Chat | "
                        f"Failed → model: {model} | "
                        f"error: {exc}"
                    )

                    continue

        if last_error:
            print(
                "PaperAxiom Chat | "
                f"All model attempts failed: {last_error}"
            )

        return None

    # ============================================================
    # CHAT WITH PAPER
    # ============================================================

    def chat_with_paper(
        self,
        question: str,
        paper_id: int,
        limit: int = 8,
    ) -> Dict:

        question = (
            question or ""
        ).strip()

        if not question:

            return {
                "answer": (
                    "Please enter a question "
                    "about the selected paper."
                ),
                "sources": [],
                "paper_id": paper_id,
            }

        # --------------------------------------------------------
        # QUERY UNDERSTANDING
        # --------------------------------------------------------

        expanded_query = self._expand_query(
            question
        )

        print(
            "PaperAxiom Chat | "
            f"Question: {question}"
        )

        # --------------------------------------------------------
        # EMBEDDING
        # --------------------------------------------------------

        try:

            query_vector = (
                embedding_service.embed_query(
                    expanded_query
                )
            )

        except Exception as exc:

            print(
                "PaperAxiom Chat | "
                f"Embedding failed: {exc}"
            )

            return {
                "answer": (
                    "I could not process the question "
                    "for this paper right now."
                ),
                "sources": [],
                "paper_id": paper_id,
            }

        # --------------------------------------------------------
        # RETRIEVAL
        # --------------------------------------------------------

        try:

            chunks = qdrant_service.search(
                query_vector=query_vector,
                paper_id=paper_id,
                limit=max(4, min(limit, 10)),
            )

        except Exception as exc:

            print(
                "PaperAxiom Chat | "
                f"Retrieval failed: {exc}"
            )

            return {
                "answer": (
                    "I could not retrieve information "
                    "from this paper right now."
                ),
                "sources": [],
                "paper_id": paper_id,
            }

        # --------------------------------------------------------
        # FILTER LOW-QUALITY RESULTS
        # --------------------------------------------------------

        valid_chunks = []

        for chunk in chunks or []:

            if not isinstance(chunk, dict):
                continue

            score = chunk.get(
                "score",
                0,
            )

            try:
                score = float(score)
            except Exception:
                score = 0

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk")
                or ""
            )

            if not str(text).strip():
                continue

            # Keep useful results.
            if score >= 0.05:
                valid_chunks.append(chunk)

        chunks = valid_chunks

        if not chunks:

            return {
                "answer": (
                    "I could not find enough relevant "
                    "information in this paper to answer "
                    "that question reliably."
                ),
                "sources": [],
                "paper_id": paper_id,
            }

        # --------------------------------------------------------
        # PAPER TITLE
        # --------------------------------------------------------

        paper_title = self._extract_paper_title(
            chunks
        )

        if not paper_title:
            paper_title = (
                "the selected research paper"
            )

        # --------------------------------------------------------
        # PREPARE EVIDENCE
        # --------------------------------------------------------

        paper_evidence = self._prepare_evidence(
            chunks,
            max_chars=7000,
        )

        if not paper_evidence:

            return {
                "answer": (
                    "Relevant information could not "
                    "be extracted from this paper."
                ),
                "sources": chunks,
                "paper_id": paper_id,
                "paper_title": paper_title,
            }

        # --------------------------------------------------------
        # FINAL RESEARCH PROMPT
        # --------------------------------------------------------

        prompt = f"""
You are answering a researcher about this paper:

PAPER TITLE:
{paper_title}

PAPER ID:
{paper_id}

RESEARCHER QUESTION:
{question}

IMPORTANT:
The researcher may phrase their question informally,
briefly, indirectly, or with imperfect wording.

First understand what they are actually asking.
Then answer the intended question.

Use the supplied paper information as the primary source.

SUPPLIED PAPER INFORMATION:
{paper_evidence}

ANSWER TASK:

Provide a clear and concise academic explanation.

Start directly with the answer.

If the question is simple, give a short answer in one
or two paragraphs.

If the question requires explanation, give a few short
paragraphs with logical flow.

Mention the paper title naturally when it helps the
researcher understand which paper is being discussed.

Explain technical terminology in simple language when
appropriate.

If numerical or technical results are present, preserve
them accurately.

If the researcher asks about something that cannot be
supported by the supplied paper information, say clearly
that the available information from the paper is
insufficient rather than guessing.

Do not mention retrieval, chunks, embeddings, Qdrant,
similarity scores, prompts, or internal system details.

Do not expose internal reasoning.

Do not produce unnecessary bullet points.

Do not repeat the question.

Do not make the answer unnecessarily long.

The final response should sound like a knowledgeable
academic researcher explaining the paper to another
researcher.
"""

        # --------------------------------------------------------
        # GENERATE
        # --------------------------------------------------------

        answer = self._call_llm(
            prompt,
            max_tokens=650,
        )

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        if answer is None:

            preview_parts = []

            for chunk in chunks[:3]:

                text = (
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("chunk")
                    or ""
                )

                text = str(text).strip()

                if text:
                    preview_parts.append(
                        text[:600]
                    )

            preview = "\n\n".join(
                preview_parts
            )

            fallback = (
                "AI synthesis is temporarily unavailable. "
                "Here is the most relevant information "
                "retrieved from the selected paper:\n\n"
                + preview[:1800]
            )

            return {
                "answer": fallback,
                "sources": chunks,
                "paper_id": paper_id,
                "paper_title": paper_title,
            }

        # --------------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------------

        return {
            "answer": answer,
            "sources": chunks,
            "paper_id": paper_id,
            "paper_title": paper_title,
        }


# ============================================================
# SINGLE SERVICE INSTANCE
# ============================================================

chat_service = ChatService()