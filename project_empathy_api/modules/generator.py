import torch
import numpy as np
import json
import re
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if not hasattr(np, "float_"):
    np.float_ = np.float64

class EmpathyLLM:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Loading Vanilla Base Model: {self.model_id} with Optimized T4 Config...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True 
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto", 
            max_memory={0: "11GiB", "cpu": "12GiB"},
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # ==========================================
    # LEGACY METHOD (Single-Pass / Toggle OFF)
    # ==========================================
    def generate(self, user_text: str, emotions: list, rag_context: list, history: list = None):
        """The original single-pass method (Ablation Baseline)."""
        if history is None: history = []
            
        try:
            emo_str = ", ".join([f"{e['label']} ({round(e['score']*100)}%)" for e in emotions[:3]])
            
            examples_list = []
            if rag_context:
                for c in rag_context:
                    ex = (f"Historical User: {c['historical_user_text']}\n"
                          f"Therapist Strategy: {c['strategy_label']}\n"
                          f"Therapist Response: {c['system_response']}")
                    examples_list.append(ex)
                examples = "\n\n".join(examples_list)
            else:
                examples = "No historical examples found. Rely on core clinical empathy."

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
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
                
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=512, min_new_tokens=225, temperature=0.85,
                    top_p=0.9, repetition_penalty=1.15, do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id
                )
                
                input_length = inputs["input_ids"].shape[1]
                decoded = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
                del inputs, output_ids
                torch.cuda.empty_cache()

            print(f"--- DEBUG: LEGACY LLM RAW OUTPUT ---\n{decoded}\n---------------------------")
            
            cleaned_decoded = re.sub(r'<[^>]*>', '', decoded).replace('assistant', '').strip()
            
            try:
                start_idx = cleaned_decoded.find('{')
                end_idx = cleaned_decoded.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = cleaned_decoded[start_idx:end_idx+1].replace('\n', ' ').replace('\r', ' ')
                    raw_data = json.loads(json_str)
                    normalized_data = {k.lower(): v for k, v in raw_data.items()}
                    if "response" in normalized_data: return normalized_data
            except Exception as e: print(f"Sanitization Error: {e}")
            
            try:
                if '"response":' in cleaned_decoded:
                    parts = cleaned_decoded.split('"response":')
                    fallback_response = parts[1].split('",')[0].replace('"', '').strip()
                    return {"chosen_strategy": "Extracted", "justification": "Malformed JSON.", "response": fallback_response}
            except: pass

            return {"chosen_strategy": "Reflection of Feelings", "justification": "Fallback: Severe formatting error.", "response": decoded.strip()}

        except Exception as e:
            traceback.print_exc()
            return {"chosen_strategy": "Error", "justification": "Internal error.", "response": "I'm having trouble processing that right now."}

    # ==========================================
    # STEP 2.5: THE BRAIN (CoT Toggle ON)
    # ==========================================
    def perform_reasoning(self, user_text: str, emotions: list, rag_context: list):
        """Analyzes context and outputs strict JSON strategy logic."""
        try:
            emo_str = ", ".join([f"{e['label']} ({round(e['score']*100)}%)" for e in emotions[:3]])
            
            examples_list = []
            if rag_context:
                for c in rag_context:
                    ex = (f"Strategy: {c['strategy_label']}\n"
                          f"Example Therapist Response: {c['system_response']}")
                    examples_list.append(ex)
                examples = "\n\n".join(examples_list)
            else:
                examples = "No historical examples found."

            system_prompt = f"""You are a Clinical Data Processor.
User Emotions: {emo_str}
Reference Strategies:
{examples}

### Your Task:
1. **Emotional Analysis**: Based on the user emotions and the user input, provide a 1 sentence summary of the user's emotional state.
2. **Strategy Selection**: Select the most appropriate counseling strategy from the Reference examples based on the user input and emotional summary.
3. **Reasoning**: Write a 1 sentence justification as to why this strategy fits their emotions.
Constraint: If the Reference Examples contain generic labels like 'Others', ignore them and select a core therapeutic strategy (e.g., Reflection of Feelings, Validation, Question) that best fits the emotion.

CRITICAL INSTRUCTIONS:
- You MUST return ONLY a valid JSON object. No other text.
- DO NOT include any conversational filler.
- ACTION-REQUEST OVERRIDE: If the user explicitly asks for advice, steps, solutions, or coping mechanisms, you MUST select 'Providing Suggestions' as your strategy.
- You MUST conclude your output with the closing bracket: }}
{{
  "emotional_analysis": "1-sentence summary of the user's core struggle",
  "chosen_strategy": "strategy name",
  "justification": "1-sentence reasoning"
}}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
            
            with torch.inference_mode():
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
                
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=200, temperature=0.3, do_sample=True, # <--- CHANGED TO 200
                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id
                )
                
                decoded = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                del inputs, output_ids
                torch.cuda.empty_cache()

            print(f"--- DEBUG STEP 2.5 (REASONING) ---\n{decoded}\n---------------------------")
            
            cleaned_decoded = re.sub(r'<[^>]*>', '', decoded).replace('assistant', '').strip()
            start_idx = cleaned_decoded.find('{')
            end_idx = cleaned_decoded.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_decoded[start_idx:end_idx+1].replace('\n', ' ').replace('\r', ' ')
                raw_data = json.loads(json_str)
                return {k.lower(): v for k, v in raw_data.items()}
                
            return {"chosen_strategy": "Reflection of Feelings", "justification": "Formatting fallback triggered."}

        except Exception as e:
            print(f"ERROR IN REASONING: {str(e)}")
            return {"chosen_strategy": "Reflection of Feelings", "justification": "Internal reasoning error."}

    # ==========================================
    # STEP 3: THE MOUTH (CoT Toggle ON)
    # ==========================================
    def generate_response(self, user_text: str, emotions: list, strategy: str, justification: str, rag_context: list, history: list = None):
        """Takes the strategy and generates a deep empathetic response."""
        if history is None: history = []
            
        try:
            emo_str = ", ".join([f"{e['label']}" for e in emotions[:3]])

            examples_list = []
            if rag_context:
                for c in rag_context:
                    ex = (f"Strategy: {c['strategy_label']}\n"
                          f"Example Therapist Response: {c['system_response']}")
                    examples_list.append(ex)
                examples = "\n\n".join(examples_list)
            else:
                examples = "No historical examples found."

            system_prompt = f"""You are a human therapist conversing directly with a client.
User Emotions: {emo_str}
Assigned Strategy: {strategy}
Strategy Logic: {justification}
Reference Strategies:
{examples}

TASK:
Write a warm, human-like, and therapeutic response (3 to 5 sentences).
- Execute the Assigned Strategy, using the Reference Examples to guide your TONE and EMPATHY.
- VOCABULARY BAN: You are strictly forbidden from starting your response with cliché therapist phrases. DO NOT use "I sense", "It sounds like", "I hear that", "I completely understand", or "I'm so sorry". 
- NATURAL OPENINGS: Start your response directly by engaging with the facts of the user's situation. Speak to them like a seasoned professional, not a textbook.
- CRITICAL CONTENT FIREWALL: Do NOT copy specific situations or life events from the Reference Examples.
- NO CRISIS ROUTING: DO NOT provide hotline numbers.
- ACTION OVERRIDE: If the Assigned Strategy is 'Providing Suggestions', you MUST immediately provide 2-3 brief, actionable steps. DO NOT defer the advice. Give the steps right now.
- QUESTION LIMIT: You must end your response with atmost ONE open-ended question. Do not include any other question marks ("?") anywhere else in your response.

OUTPUT FORMAT:
You must enclose your exact spoken response inside <response> tags. Do not write any text, notes, or explanations outside of these tags.

Example:
<response>
Carrying the weight of all those deadlines at once must be exhausting. It takes a lot of courage to admit when you are spread that thin. What is the most difficult task you are facing right now?
</response>
"""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_text})
            
            with torch.inference_mode():
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
                
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=512, min_new_tokens=90, temperature=0.8,
                    top_p=0.92, top_k=50, repetition_penalty=1.15, do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id
                )
                
                decoded = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                del inputs, output_ids
                torch.cuda.empty_cache()

            print(f"--- DEBUG STEP 3 (RAW GENERATION) ---\n{decoded}\n---------------------------")
            
            # --- THE REGEX MUZZLE ---
            import re
            
            # 1. Hunt for the tags and extract ONLY what is inside them
            match = re.search(r'<response>(.*?)</response>', decoded, re.DOTALL | re.IGNORECASE)
            
            # THE FIX: Ensure the tags actually contain text (more than 5 characters)
            if match and len(match.group(1).strip()) > 5:
                clean_text = match.group(1).strip()
            else:
                # 2. FALLBACK: If the model forgot the tags OR left them empty, strip it aggressively
                clean_text = decoded.replace('<response>', '').replace('</response>', '').strip()
                clean_text = re.split(r'(?i)\n\s*\(?Note:', clean_text)[0].strip()
                clean_text = re.split(r'(?i)\n\s*P\.S\.', clean_text)[0].strip()

            return clean_text

        except Exception as e:
            traceback.print_exc()
            print(f"ERROR IN GENERATION: {str(e)}")
            return "I'm having trouble processing that right now, but I want you to know I am here to listen."
            
    def generate_baseline(self, user_text: str, history: list = None):
        """A vanilla Llama-3 call that is intentionally concise."""
        if history is None: history = []
        try:
            messages = [{"role": "system", "content": "You are a helpful and polite AI assistant. Provide a brief response."}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_text})
            
            with torch.inference_mode():
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=250, temperature=0.7, do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id
                )
                decoded = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return decoded.strip()
        except Exception:
            return "Baseline generation failed."

empathy_llm = EmpathyLLM()