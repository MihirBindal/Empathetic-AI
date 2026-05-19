import torch
import numpy as np
import json
import re
import traceback
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

if not hasattr(np, "float_"):
    np.float_ = np.float64

class EmpathyLLM:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.adapter_path = "/content/drive/MyDrive/project_empathy_api/models"
        
        print(f"Loading Base Model and QLoRA Adapter...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True 
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto", 
            max_memory={0: "11GiB", "cpu": "12GiB"},
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        
        if os.path.exists(os.path.join(self.adapter_path, "adapter_config.json")):
            self.qlora_model = PeftModel.from_pretrained(self.model, self.adapter_path, adapter_name="empathy_adapter")
            print("QLoRA Adapter Integrated for Generation Step.")
        else:
            self.qlora_model = self.model
            print("Adapter not found. Falling back to base model.")

    # Single pass
    def generate(self, user_text: str, emotions: list, rag_context: list, history: list = None):
        if history is None: history = []
        try:
            emo_str = ", ".join([f"{e['label']} ({round(e['score']*100)}%)" for e in emotions[:3]])
            
            examples_list = []
            for c in rag_context:
                ex = (f"Historical User: {c.get('historical_user_text', 'N/A')}\n"
                      f"Therapist Strategy: {c['strategy_label']}\n"
                      f"Therapist Response: {c['system_response']}")
                examples_list.append(ex)
            examples = "\n\n".join(examples_list)

            system_prompt = f"""You are a specialized Clinical Empathetic AI.
User Emotions: {emo_str}

Reference Strategies from Database:
{examples}

### Your Task:
1. **Strategy Selection**: Select the most appropriate counseling strategy from the Reference examples.
2. **Reasoning**: Justify the choice based on the user's emotions.
3. **Professional Response**: Generate a warm, human-like, and conversational response.
    - Use the database examples for psychological guidance, but DO NOT mimic their short length.
    - Write a DEEP, ELABORATE, and comprehensive response that is atleast 8-10 sentences long.
    - Start by acknowledging the user's feelings, then explore the underlying struggle, and end with an open-ended question.
    - Avoid generic "to-do list" advice unless using a 'Suggestions' strategy.

CRITICAL INSTRUCTIONS:
- You are a backend data processor.
- You MUST return ONLY a valid JSON object.
- DO NOT include any conversational filler or "Here is the output" notes outside the JSON.

Return ONLY this exact JSON format:
{{
  "chosen_strategy": "strategy name",
  "justification": "reasoning",
  "response": "your empathetic reply"
}}"""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_text})
            
            with torch.inference_mode():
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(formatted, return_tensors="pt").to("cuda")
                with self.qlora_model.disable_adapter():
                    output_ids = self.model.generate(
                        input_ids=inputs["input_ids"], max_new_tokens=512, min_new_tokens=225, 
                        temperature=0.85, top_p=0.9, repetition_penalty=1.15,
                        do_sample=True, pad_token_id=self.tokenizer.eos_token_id
                    )
                decoded = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                torch.cuda.empty_cache()

            print(f"--- DEBUG: LEGACY LLM RAW OUTPUT ---\n{decoded}\n---------------------------")
            match = re.search(r'\{.*\}', decoded, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"chosen_strategy": "Reflection of Feelings", "justification": "JSON Parse Error", "response": decoded.strip()}
        except Exception as e:
            print(f"LEGACY CRASH: {e}")
            return {"chosen_strategy": "Error", "justification": "Error.", "response": "I'm having trouble processing that right now."}

    # CoT reasoning
    def perform_reasoning(self, user_text, emotions, rag_context):
        try:
            emo_str = ", ".join([f"{e['label']} ({round(e['score']*100)}%)" for e in emotions[:3]])
            examples = "\n\n".join([f"Strategy: {c.get('strategy_label', 'Unknown')}\nExample: {c.get('system_response', 'N/A')}" for c in rag_context])
            
            system_prompt = f"""You are an expert Clinical Psychology Data Processor.
User Emotions: {emo_str}

Available Reference Strategies:
{examples}

### Your Task:
You must analyze the user's input and select the absolute best therapeutic strategy from the Available Reference Strategies. 

CRITICAL INSTRUCTIONS:
- You MUST return ONLY a valid JSON object. No other text.
- ACTION-REQUEST OVERRIDE: If the user explicitly asks for advice, steps, or solutions, you MUST select 'Providing Suggestions'.
- You MUST conclude your output with the closing bracket: }}

Return EXACTLY this JSON format:
{{
  "emotional_analysis": "Identify the primary emotion here.",
  "user_situation_summary": "1-sentence summary of the user's situation.",
  "chosen_strategy": "Exact name of the strategy selected from the references.",
  "justification": "Exactly ONE concise sentence explaining why this strategy fits the user's situation."
}}"""

            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": f"USER INPUT TO ANALYZE: {user_text}"}
            ]
            
            with torch.inference_mode():
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                formatted += '{\n  "emotional_analysis": "'
                
                inputs = self.tokenizer(formatted, return_tensors="pt").to("cuda")
                
                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                with self.qlora_model.disable_adapter():
                    output_ids = self.model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"], 
                        max_new_tokens=250, 
                        temperature=0.2, 
                        repetition_penalty=1.15,
                        do_sample=True,
                        eos_token_id=terminators,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                input_length = inputs["input_ids"].shape[1]
                decoded = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
                
                full_json_string = '{\n  "emotional_analysis": "' + decoded
                torch.cuda.empty_cache()

            print(f"--- DEBUG STEP 3 (REASONING) ---\n{full_json_string}\n---------------------------")
            
            import re
            import json
            match = re.search(r'\{.*\}', full_json_string, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"chosen_strategy": "Reflection of Feelings", "justification": "Reasoning failed JSON formatting."}
        except Exception as e:
            print(f"REASONING CRASH: {e}")
            return {"chosen_strategy": "Reflection of Feelings", "justification": "Internal error."}


    # CoT generator
    def generate_response(self, user_text: str, emotions: list, strategy: str, justification: str, rag_context: list, history: list = None, **kwargs) -> str:
        print("--- DEBUG STEP 4 (QLoRA GENERATION) ---")
        import re
        
        emo_str = ", ".join([e["label"] for e in emotions])
        
        system_prompt = "You are an empathetic, professional therapist. You will be given the user's message, their detected emotional state, a recommended therapeutic strategy, and the justification for that strategy. Generate a response that follows the strategy and addresses the user's emotional state directly."

        user_content = f"""User message: {user_text}
Detected emotions: {emo_str}
Strategy: {strategy}
Justification: {justification}

CRITICAL EXECUTION CHECKLIST:
1. STRATEGY LOCK: You are executing '{strategy}'. If '{strategy}' is NOT 'Providing Suggestions', you are FORBIDDEN from giving tips, advice, or solutions.
2. NO INTERROGATION: You may ask EXACTLY ONE question at the very end of your response. Zero questions are allowed before the end.
3. NO CLICHES: Do NOT start with "It sounds like" or "I hear that".
4. NO LISTS: Write in a natural, warm paragraph. Do not use bullet points.
5. NO REFERRALS: Do not tell the user to seek a counselor or talk to a teacher. You are their counselor right now.
6. FORMAT: You MUST enclose your response inside <response> tags."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        with torch.inference_mode():
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            formatted_prompt += "<response>\n"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], 
                    max_new_tokens=150,
                    temperature=0.4, 
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.15,
                    do_sample=True,
                    eos_token_id=terminators,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                input_length = inputs["input_ids"].shape[1]
                response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
                
                print(response_text)
                print("---------------------------")
                
                clean_text = response_text.replace("</response>", "").replace("<response>", "").strip()
                return clean_text

            except Exception as e:
                print(f"Error in Step 4: {e}")
                return "I'm here to listen. Can you tell me a little more about what's going on?"

    def generate_baseline(self, user_text, history=None):
        if history is None: history = []
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": user_text}]
        with torch.inference_mode():
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted, return_tensors="pt").to("cuda")
            with self.qlora_model.disable_adapter():
                output_ids = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=200, temperature=0.7)
            return self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

empathy_llm = EmpathyLLM()