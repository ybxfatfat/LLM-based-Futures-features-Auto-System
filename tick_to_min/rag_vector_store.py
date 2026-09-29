# rag_vector_store.py
import os
import json
import numpy as np
from openai import OpenAI

class LocalFactorKB:
    def __init__(self, db_path="./data/factor_kb.json"):
        self.db_path = db_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.kb_data = []
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.kb_data = json.load(f)

    def _save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.kb_data, f, ensure_ascii=False, indent=2)

    def get_embedding(self, text: str) -> list:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def add_experience(self, feature_name: str, hypothesis: dict, code: str, metrics: dict, reflection: str):
        """记录完整的特征生命周期，仅对 hypothesis 进行 embedding"""
        text_to_embed = f"{feature_name}: {hypothesis.get('mechanism', '')} {hypothesis.get('counterparty_loser', '')}"
        embedding = self.get_embedding(text_to_embed)
        
        record = {
            "feature_name": feature_name,
            "hypothesis": hypothesis,
            "python_class_code": code,
            "metrics": metrics,
            "reflection": reflection,
            "embedding": embedding
        }
        self.kb_data.append(record)
        self._save_db()
        print(f"💾 特征 [{feature_name}] 的反思与代码已成功录入 RAG 向量知识库！")

    def query_similar_experience(self, idea_text: str, top_k: int = 3, threshold: float = 0.5) -> list:
        if not self.kb_data:
            return []
            
        query_emb = np.array(self.get_embedding(idea_text))
        
        results = []
        for item in self.kb_data:
            doc_emb = np.array(item["embedding"])
            # 计算 Cosine Similarity
            cosine_sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            if cosine_sim >= threshold:
                results.append((cosine_sim, item))
                
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:top_k]]