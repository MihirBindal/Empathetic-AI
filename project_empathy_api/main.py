import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import traceback

from modules.safety import gatekeeper
from modules.classifier import emotion_classifier
from modules.rag_engine import rag_db
from modules.generator import empathy_llm

app = FastAPI(title="Project Empathy v3.0 Orchestrator")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_text: str
    history: Optional[List[ChatMessage]] = []
    run_baseline_comparison: bool = True
    enable_cot: bool = True  # <--- NEW: The Toggle Parameter

@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # STEP 0 & 1: Safety and Emotion
        safety_check = gatekeeper.check_safety(request.user_text)
        if not safety_check["is_safe"]:
            return {
                "status": "crisis_detected",
                "metadata": {"action": safety_check["action"]},
                "results": {"project_empathy_output": safety_check["emergency_payload"]}
            }

        emotions = emotion_classifier.predict(request.user_text)
        rag_context = rag_db.retrieve(request.user_text, emotions)
        formatted_history = [{"role": m.role, "content": m.content} for m in request.history]

        # =========================================================
        # THE ABLATION SWITCH (CoT vs Single-Pass)
        # =========================================================
        if request.enable_cot:
            # TRUE V3.0 PIPELINE (Two-Step CoT)
            reasoning_data = empathy_llm.perform_reasoning(
                user_text=request.user_text, emotions=emotions, rag_context=rag_context
            )
            chosen_strategy = reasoning_data.get("chosen_strategy", "Reflective Listening")
            justification = reasoning_data.get("justification", "Default emotional alignment.")

            project_empathy_reply = empathy_llm.generate_response(
                user_text=request.user_text, 
                emotions=emotions, 
                strategy=chosen_strategy, 
                justification=justification,
                rag_context=rag_context, 
                history=formatted_history
            )
        else:
            # LEGACY PIPELINE (Single-Pass Hacky JSON)
            empathy_data = empathy_llm.generate(
                user_text=request.user_text, emotions=emotions, 
                rag_context=rag_context, history=formatted_history
            )
            chosen_strategy = empathy_data.get("chosen_strategy", "Unknown")
            justification = empathy_data.get("justification", "No justification provided.")
            project_empathy_reply = empathy_data.get("response", "")

        # STEP 3b: Baseline
        baseline_output = "Baseline disabled to save compute."
        if request.run_baseline_comparison:
            baseline_output = empathy_llm.generate_baseline(request.user_text, formatted_history)

        return {
            "status": "success",
            "metadata": {
                "detected_emotions": emotions,
                "strategy_selected": chosen_strategy,
                "strategy_justification": justification,
                "cot_enabled": request.enable_cot  # Good for debugging
            },
            "results": {
                "project_empathy_output": project_empathy_reply,
                "standard_llama_output": baseline_output
            }
        }

    except Exception as e:
        print("CRASH IN ORCHESTRATOR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))