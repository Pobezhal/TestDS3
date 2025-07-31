import threading
from collections import deque
from typing import List, Any
from uuid import uuid4

class ChatMemoryManager:
    """
    Manages long-term conversational memory:
      - raw line storage in Chroma
      - verbatim history tail (capped)
      - similarity-based snippet retrieval
    """
    def __init__(
        self,
        chat_id: str,
        user_id: str,
        chroma_collection: Any,
        embedder: Any,
        openai_client: Any,
        memory_n_verbatim: int = 24,
    ):
        # Identifiers
        self.chat_id = chat_id
        self.user_id = user_id

        # External interfaces
        self.chroma = chroma_collection
        self.embedder = embedder
        self.openai = openai_client

        # Memory parameters
        self.memory_n_verbatim = memory_n_verbatim

        # Internal state
        self.verbatim_queue = deque(maxlen=memory_n_verbatim)
        self._lock = threading.Lock()

    def add_message(self, role: str, text: str) -> None:
        with self._lock:
            # 1) Enqueue verbatim
            self.verbatim_queue.append({"role": role, "text": text})

            # 2) Check existing message count for this (chat_id, user_id)
            #    and enforce max 600 messages
            MAX_MESSAGES = 600

            results = self.chroma.get(
                where={
                    "$and": [
                        {"chat_id": {"$eq": str(self.chat_id)}},
                        {"user_id": {"$eq": str(self.user_id)}},
                        {"type": {"$eq": "line"}}
                    ]
                },
                include=["metadatas"]
            )
            message_ids = results["ids"]
            current_count = len(results["metadatas"])

            # If we already have 600 or more messages, delete the oldest 10
            if current_count >= MAX_MESSAGES:
                ids_to_delete = message_ids[:10]  # Delete oldest 10 to reduce pressure
                self.chroma.delete(ids=ids_to_delete)

            # 3) Add new message to Chroma
            embedding = self.embedder.encode_documents([text])[0]
            metadata = {
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "type": "line",
                "role": role,
            }
            doc_id = str(uuid4())
            self.chroma.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )

    def build_prompt_parts(self, user_query: str) -> List[str]:
        """
        Construct prompt blocks:
          1. Verbatim tail (last 3500 chars of raw history)
          2. Similar snippets (up to 1500 chars from Chroma)
        """
        with self._lock:
            parts: List[str] = []

            # 1) Verbatim tail: join the in-memory queue and cap to 3000 chars
            full_history = "\n".join(
                f"{m['role']}: {m['text']}" for m in self.verbatim_queue
            )
            parts.append(full_history[-3500:])

            # 2) Retrieve up to 1200 chars of top-k similar lines
            query_emb = self.embedder.encode_queries([user_query])[0]
            results = self.chroma.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"$and": [{"chat_id": {"$eq": str(self.chat_id)}}, {"user_id": {"$eq": str(self.user_id)}}]},
            )
            snippet_buf: List[str] = []
            total_len = 0
            for doc in results.get("documents", [[]])[0]:
                doc_len = len(doc)
                if total_len + doc_len > 1500:
                    break
                snippet_buf.append(doc)
                total_len += doc_len
            parts.append("\n".join(snippet_buf))

            return parts
