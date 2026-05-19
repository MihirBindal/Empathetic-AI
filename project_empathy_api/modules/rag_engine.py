import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# --- STEP 2.5 COMPATIBILITY PATCH ---
if not hasattr(np, "float_"):
    np.float_ = np.float64

# --- THE TAXONOMY MAP ---
# Maps RoBERTa's 28 granular emotions to ESConv's 11 core categories
TAXONOMY_MAP = {
    "anger": "anger", "anxiety": "anxiety", "disgust": "disgust", 
    "fear": "fear", "jealousy": "jealousy", "nervousness": "nervousness", 
    "sadness": "sadness", 
    "annoyance": "anger", "disapproval": "disgust", "embarrassment": "shame",
    "grief": "sadness", "remorse": "guilt", "disappointment": "depression",
    "confusion": "anxiety",
    "joy": None, "neutral": None, "caring": None, "admiration": None, 
    "amusement": None, "approval": None, "curiosity": None, "desire": None, 
    "excitement": None, "gratitude": None, "love": None, "optimism": None, 
    "pride": None, "realization": None, "relief": None, "surprise": None
}

class RAGEngine:
    def __init__(self):
        db_path = "/content/drive/MyDrive/project_empathy_api/models/chroma_db"
        print(f"Loading Hybrid RAG Engine at {db_path}...")
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "esconv_strategies"
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} 
        )

    def _get_embedding(self, text: str) -> list:
        return self.embedder.encode(text).tolist()

    def retrieve(self, user_text: str, emotions: list, top_k: int = 5) -> list:
        """
        HYBRID RAG: Metadata Pre-Filtering + Weighted Semantic Re-Ranking
        """
        # 1. Extract Top 3 Emotions from the classifier
        top_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]
        
        # 2. Map GoEmotions to ESConv Taxonomy
        target_emotions = set()
        for emp in top_emotions:
            mapped_type = TAXONOMY_MAP.get(emp['label'])
            if mapped_type:
                target_emotions.add(mapped_type)
                
        # 3. Build the Hard Filter (The Rigid Wall)
        where_filter = None 
        if len(target_emotions) == 1:
            where_filter = {"emotion_type": list(target_emotions)[0]}
        elif len(target_emotions) > 1:
            where_filter = {"$or": [{"emotion_type": e} for e in target_emotions]}

        all_candidates = {}

        # 4. Parallel Query Execution
        for emotion in top_emotions:
            emo_label = emotion['label']
            emo_weight = emotion['score']
            
            query_string = f"[{emo_label}] {user_text}"
            query_embedding = self._get_embedding(query_string)
            
            try:
                # 5. Execute search inside the isolated emotional cluster
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter 
                )
            except Exception as e:
                # Triggers if the filter maps to a category that happens to be empty
                continue
            
            # 6. Max-Pooling and Confidence Weighting
            if results and "documents" in results and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    raw_distance = results["distances"][0][i] if "distances" in results else 1.0
                    
                    base_similarity = 1.0 - raw_distance
                    weighted_score = base_similarity * emo_weight
                    
                    if doc not in all_candidates or weighted_score > all_candidates[doc]['weighted_score']:
                        all_candidates[doc] = {
                            "historical_user_text": doc,
                            "matched_emotion": meta.get("emotion_type", "Unknown"),
                            "problem_type": meta.get("problem_type", "Unknown"),
                            
                            # --- ADD THESE TWO LINES BACK ---
                            # This provides the keys generator.py is looking for, 
                            # with a safe fallback so the API never crashes again.
                            "strategy_label": meta.get("strategy", "Reflective Listening"),
                            "system_response": meta.get("response", "I hear you. Tell me more about how that feels."),
                            
                            "weighted_score": weighted_score
                        }

        # 7. Final Output Rank
        ranked_results = sorted(all_candidates.values(), key=lambda x: x['weighted_score'], reverse=True)
        return ranked_results[:top_k]

# Initialize the engine
rag_db = RAGEngine()