import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SafetyGatekeeper:
    def __init__(self):
        # Swapping to a fine-tuned crisis detection model
        model_name = "sentinet/suicidality"
        
        print(f"Loading Safety Gatekeeper: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.crisis_threshold = 0.98
        

    def check_safety(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        # For 'sentinet/suicidality', index 1 corresponds to crisis/suicidality
        crisis_prob = probabilities[0][1].item() 
        
        if crisis_prob >= self.crisis_threshold:
            return {
                "is_safe": False,
                "confidence": round(crisis_prob, 4),
                "action": "ABORT_PIPELINE",
                "emergency_payload": "It takes a lot of courage to share that you are feeling this way. I am an AI and cannot offer the crisis support you need and deserve right now. Your safety is the most important thing, and you do not have to carry this heavy burden alone. Please reach out to a professional who can help immediately. You can call the Kiran mental health helpline at 1800-599-0019, or the Vandrevala Foundation at 9999 666 555. Please connect with them—your life matters."
            }
            
        return {
            "is_safe": True,
            "confidence": round(1.0 - crisis_prob, 4),
            "action": "PROCEED"
        }

gatekeeper = SafetyGatekeeper()