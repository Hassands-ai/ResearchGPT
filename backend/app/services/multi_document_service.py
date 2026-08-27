from typing import Dict, List

from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service


class MultiDocumentService:
    """
    Retrieves research evidence independently for each paper.

    The service:
    - searches each paper separately
    - keeps evidence associated with the correct paper
    - removes duplicate evidence
    - uses a fallback query when the primary query gives
      insufficient evidence
    - returns the strongest evidence first
    """

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    def _normalize_text(self, text: str) -> str:

        if not text:
            return ""

        text = text.lower()

        text = (
            text
            .replace("\n", " ")
            .replace("\r", " ")
        )

        return " ".join(
            text.split()
        )

    # =========================================================
    # DUPLICATE DETECTION
    # =========================================================

    def _is_duplicate(
        self,
        text: str,
        existing_texts: List[str],
    ) -> bool:

        normalized = self._normalize_text(
            text
        )

        if not normalized:
            return True

        words_a = normalized.split()

        for existing in existing_texts:

            existing_normalized = (
                self._normalize_text(
                    existing
                )
            )

            if not existing_normalized:
                continue

            # Exact duplicate
            if normalized == existing_normalized:
                return True

            words_b = existing_normalized.split()

            if (
                len(words_a) < 30
                or len(words_b) < 30
            ):
                continue

            # Compare first 100 words
            sample_a = set(
                words_a[:100]
            )

            sample_b = set(
                words_b[:100]
            )

            union = sample_a.union(
                sample_b
            )

            if not union:
                continue

            intersection = sample_a.intersection(
                sample_b
            )

            similarity = (
                len(intersection)
                / len(union)
            )

            if similarity >= 0.90:
                return True

        return False

    # =========================================================
    # CLEAN RETRIEVED TEXT
    # =========================================================

    def _clean_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = str(text)

        text = (
            text
            .replace("\r", " ")
            .replace("\n", " ")
        )

        # Normalize whitespace
        text = " ".join(
            text.split()
        )

        return text.strip()

    # =========================================================
    # SEARCH ONE PAPER
    # =========================================================

    def _search_one_paper(
        self,
        query_vector,
        paper_id: int,
        limit: int,
    ) -> List[Dict]:

        try:

            results = qdrant_service.search(
                query_vector=query_vector,
                paper_id=paper_id,
                limit=max(
                    limit * 3,
                    10,
                ),
            )

        except Exception as exc:

            print(
                f"Qdrant search failed "
                f"| paper={paper_id} "
                f"| error={exc}"
            )

            return []

        paper_results = []

        existing_texts = []

        for result in results:

            # -------------------------------------------------
            # HARD PAPER-ID CHECK
            # -------------------------------------------------

            result_paper_id = result.get(
                "paper_id"
            )

            if result_paper_id is not None:

                try:

                    if int(result_paper_id) != int(
                        paper_id
                    ):
                        continue

                except Exception:
                    continue

            text = self._clean_text(
                result.get("text") or ""
            )

            if not text:
                continue

            # Ignore extremely small fragments
            if len(text) < 60:
                continue

            if self._is_duplicate(
                text,
                existing_texts,
            ):
                continue

            score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            existing_texts.append(
                text
            )

            paper_results.append(
                {
                    "paper_id": paper_id,
                    "text": text,
                    "score": score,
                }
            )

            if len(paper_results) >= limit:
                break

        return paper_results

    # =========================================================
    # MAIN SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        paper_ids: List[int],
        limit_per_paper: int = 5,
    ) -> Dict:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty"
            )

        if not paper_ids:

            raise ValueError(
                "At least one paper_id is required"
            )

        if limit_per_paper < 1:
            limit_per_paper = 1

        # Remove duplicate paper IDs
        unique_paper_ids = list(
            dict.fromkeys(
                paper_ids
            )
        )

        clean_query = query.strip()

        # =====================================================
        # PRIMARY QUERY EMBEDDING
        # =====================================================

        try:

            query_vector = (
                embedding_service.embed_query(
                    clean_query
                )
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not create query embedding: {exc}"
            )

        all_results = []

        # =====================================================
        # SEARCH EACH PAPER INDEPENDENTLY
        # =====================================================

        for paper_id in unique_paper_ids:

            paper_results = (
                self._search_one_paper(
                    query_vector=query_vector,
                    paper_id=paper_id,
                    limit=limit_per_paper,
                )
            )

            # -------------------------------------------------
            # FALLBACK SEARCH
            #
            # If the first semantic query does not return
            # enough evidence, try a shorter/general query.
            # -------------------------------------------------

            if len(paper_results) < min(
                2,
                limit_per_paper,
            ):

                fallback_query = (
                    "research paper "
                    "methodology results "
                    "dataset model contribution "
                    "experiment"
                )

                try:

                    fallback_vector = (
                        embedding_service.embed_query(
                            fallback_query
                        )
                    )

                    fallback_results = (
                        self._search_one_paper(
                            query_vector=fallback_vector,
                            paper_id=paper_id,
                            limit=limit_per_paper * 2,
                        )
                    )

                except Exception as exc:

                    print(
                        f"Fallback search failed "
                        f"| paper={paper_id} "
                        f"| error={exc}"
                    )

                    fallback_results = []

                # -------------------------------------------------
                # Merge primary + fallback evidence
                # -------------------------------------------------

                merged = []

                seen_texts = []

                for item in (
                    paper_results
                    + fallback_results
                ):

                    text = item.get(
                        "text",
                        "",
                    )

                    if not text:
                        continue

                    if self._is_duplicate(
                        text,
                        seen_texts,
                    ):
                        continue

                    seen_texts.append(
                        text
                    )

                    merged.append(
                        item
                    )

                    if len(merged) >= limit_per_paper:
                        break

                paper_results = merged

            # -------------------------------------------------
            # ADD PAPER RESULTS
            # -------------------------------------------------

            all_results.extend(
                paper_results
            )

        # =====================================================
        # SORT
        # =====================================================

        all_results.sort(
            key=lambda item: (
                item["paper_id"],
                -item["score"],
            )
        )

        # =====================================================
        # DEBUG INFORMATION
        # =====================================================

        for paper_id in unique_paper_ids:

            count = sum(
                1
                for item in all_results
                if item["paper_id"] == paper_id
            )

            print(
                f"Multi-document evidence "
                f"| Paper {paper_id} "
                f"| Results: {count}"
            )

        # =====================================================
        # RESPONSE
        # =====================================================

        return {
            "query": clean_query,
            "paper_ids": unique_paper_ids,
            "papers_count": len(
                unique_paper_ids
            ),
            "results_count": len(
                all_results
            ),
            "results": all_results,
        }


multi_document_service = MultiDocumentService()