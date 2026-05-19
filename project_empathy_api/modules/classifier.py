import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class EmotionClassifier:
    def __init__(self):
        model_path = "SamLowe/roberta-base-go_emotions"
        print(f"Loading Emotion Classifier: {model_path}...")
        self.device = torch.device("cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)
            
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label = self.model.config.id2label[idx.item()]
            results.append({
                "label": label,
                "score": round(prob.item(), 4) 
            })
            
        return results

emotion_classifier = EmotionClassifier()