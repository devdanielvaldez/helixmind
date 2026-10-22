"""Vector indexing and semantic RAG engine using Ollama embeddings."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
import numpy as np

from helixmind.llm.ollama_client import OllamaClient
from helixmind.papers.models import ClinicalStudy, DocumentChunk, Paper


class SemanticIndex:
    """In-memory fast vector index for scientific literature and clinical trials."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama = ollama_client or OllamaClient()
        self.chunks: List[DocumentChunk] = []
        self.matrix: Optional[np.ndarray] = None  # shape (N, D)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def clear(self) -> None:
        self.chunks.clear()
        self.matrix = None

    def chunk_paper(self, paper: Paper, max_chunk_chars: int = 700) -> List[DocumentChunk]:
        """Split a paper into logical semantic chunks."""
        chunks: List[DocumentChunk] = []

        # 1. Title chunk
        chunks.append(
            DocumentChunk(
                chunk_id=f"{paper.id}_title",
                doc_id=paper.id,
                doc_title=paper.title,
                source_type="paper",
                section="title",
                content=f"Title: {paper.title} (Journal: {paper.journal or 'N/A'}, Year: {paper.pub_year or 'N/A'})",
            )
        )

        # 2. Abstract paragraphs/sentences
        raw_abstract = paper.abstract.strip()
        if raw_abstract and raw_abstract != paper.title:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\.\s+(?=[A-Z])", raw_abstract) if len(p.strip()) > 30]
            current_chunk = ""
            for i, p in enumerate(paragraphs):
                if len(current_chunk) + len(p) < max_chunk_chars:
                    current_chunk += " " + p
                else:
                    if current_chunk.strip():
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{paper.id}_abs_{len(chunks)}",
                                doc_id=paper.id,
                                doc_title=paper.title,
                                source_type="paper",
                                section="abstract",
                                content=current_chunk.strip(),
                            )
                        )
                    current_chunk = p

            if current_chunk.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{paper.id}_abs_{len(chunks)}",
                        doc_id=paper.id,
                        doc_title=paper.title,
                        source_type="paper",
                        section="abstract",
                        content=current_chunk.strip(),
                    )
                )

        return chunks

    def chunk_clinical_study(self, study: ClinicalStudy) -> List[DocumentChunk]:
        """Split a clinical study into semantic chunks."""
        chunks: List[DocumentChunk] = []

        summary_text = (
            f"Clinical Trial {study.nct_id}: {study.title}\n"
            f"Status: {study.status} | Phase: {study.phase or 'N/A'} | Sponsor: {study.lead_sponsor or 'N/A'}\n"
            f"Conditions: {', '.join(study.conditions)}\n"
            f"Interventions: {', '.join(study.interventions)}\n"
            f"Summary: {study.brief_summary}"
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"{study.nct_id}_summary",
                doc_id=study.nct_id,
                doc_title=study.title,
                source_type="trial",
                section="protocol",
                content=summary_text,
            )
        )
        return chunks

    async def index_documents_async(
        self,
        papers: List[Paper],
        studies: Optional[List[ClinicalStudy]] = None,
        batch_size: int = 16,
    ) -> int:
        """Extract chunks from papers and studies, generate embeddings, and build the index."""
        all_new_chunks: List[DocumentChunk] = []

        for p in papers:
            all_new_chunks.extend(self.chunk_paper(p))

        if studies:
            for s in studies:
                all_new_chunks.extend(self.chunk_clinical_study(s))

        if not all_new_chunks:
            return 0

        # Generate embeddings in batches via Ollama with resource protection
        from helixmind.core.resources import get_resource_manager
        rm = get_resource_manager()

        texts = [c.content for c in all_new_chunks]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb_batch = await self.ollama.embed_async(batch)
            all_embeddings.extend(emb_batch)
            await rm.throttle_if_needed()

        for chunk, emb in zip(all_new_chunks, all_embeddings):
            chunk.embedding = emb

        # Retain chunks with valid embeddings
        valid_chunks = [c for c in all_new_chunks if c.embedding and len(c.embedding) > 0]
        if not valid_chunks:
            return 0

        self.chunks.extend(valid_chunks)

        # Build / update numpy matrix
        new_vecs = np.array([c.embedding for c in valid_chunks], dtype=np.float32)
        # Normalize for fast cosine dot-product
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_vecs = new_vecs / norms

        if self.matrix is None:
            self.matrix = norm_vecs
        else:
            self.matrix = np.vstack([self.matrix, norm_vecs])

        return len(valid_chunks)

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.2,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Semantic search against indexed chunks using cosine similarity."""
        if self.is_empty() or self.matrix is None:
            return []

        q_embs = await self.ollama.embed_async(query)
        if not q_embs or len(q_embs[0]) == 0:
            return []

        q_vec = np.array(q_embs[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        # Cosine similarity via dot product against normalized matrix
        scores = np.dot(self.matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.chunks[idx], score))

        return results

    async def get_relevant_context_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve relevant chunks and format into context for prompt engineering."""
        matches = await self.search_async(query, top_k=top_k)
        if not matches:
            return "No matching scientific literature found in the index."

        context_parts = []
        for i, (chunk, score) in enumerate(matches, 1):
            context_parts.append(
                f"[{i}] [{chunk.source_type.upper()}] {chunk.doc_title} ({chunk.doc_id}) (Relevance: {score:.2f}):\n"
                f"{chunk.content}"
            )

        return "\n\n".join(context_parts)
