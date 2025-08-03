import threading
from collections import deque
from typing import List, Any
from uuid import uuid4
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class ChatMemoryManager:
    """
    Manages long-term conversational memory:
      - raw line storage in Chroma
      - verbatim history tail (capped)
      - similarity-based snippet retrieval
    """
    TRIGGERS = ["remember", "in past", "previously", "remind", "recall", "вспомни", "напомни", "ранее", "до этого", "помнишь", "мы говорили", "вчера", "позавчера"]
    def __init__(
        self,
        chat_id: str,
        user_id: str,
        chroma_collection: Any,
        embedder: Any,
        openai_client: Any,
        memory_n_verbatim: int = 40,
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
        self._write_counter = 0
        self.WRITE_BATCH = 30

    def add_message(self, role: str, text: str) -> None:
        with self._lock:
            timestamp = datetime.utcnow().isoformat()
            # 1) Always keep in verbatim queue
            self.verbatim_queue.append({"role": role, "text": text, "timestamp": timestamp})
    
            # 2) Increment counter
            self._write_counter += 1
    
            # 3) Only batch-write every 30 messages
            if self._write_counter % self.WRITE_BATCH != 0:
                return
    
            # 4) Get the oldest 25 messages (not the newest)
            #    Skip the last 10 so we don't lose recent ones
            batch_to_store = list(self.verbatim_queue)[10:]  # Oldest 30 (if maxlen=40)
    
            if not batch_to_store:
                return
    
            try:
                texts = [m["text"] for m in batch_to_store]
                roles = [m["role"] for m in batch_to_store]
                timestamps = [m["timestamp"] for m in batch_to_store]
    
                embeddings = self.embedder.embed_documents(texts)
    
                ids = [str(uuid4()) for _ in texts]
                metadatas = [
                    {
                        "chat_id": str(self.chat_id),
                        "user_id": str(self.user_id),
                        "type": "line",
                        "role": role,
                        "timestamp": timestamp,
                    }
                    for role, timestamp in zip(roles, timestamps)
                ]
    
                self.chroma.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.warning(f"Chroma batch add failed: {e}")

    
    def build_prompt_parts(self, user_query: str) -> List[str]:
        """
        Construct prompt blocks:
          1. Verbatim tail (last 3500 chars of raw history)
          2. Similar snippets (up to 1200 chars from Chroma)
        """
        with self._lock:
            parts: List[str] = []

            # 1) Verbatim tail: join the in-memory queue and cap to 3500 chars
            full_history = "\n".join(
                f"{m['role']}: {m['text']}" for m in self.verbatim_queue
            )
            parts.append(f"ИСТОРИЯ:\n{full_history[-3500:]}")

            if any(trigger in user_query.lower() for trigger in self.TRIGGERS):
                # 2) Retrieve up to 1200 chars of top-k similar lines
                query_emb = self.embedder.embed_query(user_query)
                results = self.chroma.query(
                    query_embeddings=[query_emb],
                    n_results=4,
                    where={"$and": [{"chat_id": {"$eq": str(self.chat_id)}}, {"user_id": {"$eq": str(self.user_id)}}]},
                )
                snippet_buf: List[str] = []
                total_len = 0
                for doc in results.get("documents", [[]])[0]:
                    doc_len = len(doc)
                    if total_len + doc_len > 1200:
                        break
                    snippet_buf.append(doc)
                    total_len += doc_len
                if snippet_buf:
                    parts.append("ПОХОЖИЕ СООБЩЕНИЯ:\n" + "\n".join(snippet_buf))
    
            return parts

