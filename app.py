import streamlit as st
import requests

API_URL = "https://ranular-unhoroscopic-jonas.ngrok-free.dev/generate" 

st.set_page_config(page_title="Solace AI", page_icon="", layout="wide")

st.markdown("""
    <style>
    /* 1. Remove Streamlit's massive default top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 95%;
    }
    
    /* 2. Main Canvas Background - PITCH BLACK */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }
    
    /* 3. The Cards - DISTINCT GREY to pop out */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1E1E1E !important; 
        border: 1px solid #404040 !important;
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* Scrollbar styling for Analysis Card */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        scrollbar-width: thin;
        scrollbar-color: #404040 #1E1E1E;
    }
    
    /* Center the toggles vertically */
    .stToggle {
        padding-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_metadata" not in st.session_state:
    st.session_state.latest_metadata = None
if "latest_baseline" not in st.session_state:
    st.session_state.latest_baseline = None

header_left, header_right = st.columns([2.2, 1], gap="medium")

with header_left:
    st.markdown("### Solace AI")

with header_right:
    col1, col2, col3 = st.columns([1, 1, 1]) 
    with col1:
        run_baseline = st.toggle("Baseline", value=True)
    with col2:
        enable_cot = st.toggle("CoT Split", value=True) 
    with col3:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.latest_metadata = None
            st.session_state.latest_baseline = None
            st.rerun()

left_col, right_col = st.columns([2.2, 1], gap="medium")

with left_col:
    # Restored to 480 height to match your preferred layout
    chat_card = st.container(height=480, border=True)
    
    with chat_card:
        if len(st.session_state.messages) == 0:
            st.caption("Start a conversation...")
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("How are you feeling today?")

with right_col:
    analysis_card = st.container(height=550, border=True) 
    
    with analysis_card:
        if st.session_state.latest_metadata:
            meta = st.session_state.latest_metadata
            
            emo_display = "N/A"
            if "detected_emotions" in meta:
                emo_strings = [f"{e['label'].title()} ({round(e['score'] * 100)}%)" for e in meta["detected_emotions"]]
                emo_display = " • ".join(emo_strings)
            
            strategy = meta.get('strategy_selected', 'N/A')
            reasoning = meta.get('strategy_justification', 'No justification.')
            baseline = st.session_state.latest_baseline
            
            html_content = f"""<div style="font-size: 15px; line-height: 1.5; font-family: sans-serif;">
<div style="margin-bottom: 6px;"><strong>Top Emotions:</strong><br><span style="color: #E0E0E0;">{emo_display}</span></div>
<hr style="margin: 12px 0; border: none; border-top: 1px solid #404040;">
<div style="margin-bottom: 6px;"><strong>Strategy Used:</strong><br><span style="color: #E0E0E0;">{strategy}</span></div>
<hr style="margin: 12px 0; border: none; border-top: 1px solid #404040;">
<div style="margin-bottom: 4px;"><strong>Reasoning:</strong></div>
<div style="color: #A0AAB4; font-size: 14px; margin-bottom: 10px;">{reasoning}</div>
"""
            if baseline:
                html_content += f"""<hr style="margin: 12px 0; border: none; border-top: 1px solid #404040;">
<div style="margin-bottom: 4px;"><strong>Baseline:</strong></div>
<div style="color: #A0AAB4; font-size: 14px; line-height: 1.6;">{baseline}</div>
"""
            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)
            
        else:
            st.markdown("<div style='height: 100%; display: flex; align-items: center; justify-content: center; color: #555;'>Start a conversation to see the analysis...</div>", unsafe_allow_html=True)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_card:
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing emotional context..."):
                
                payload = {
                    "user_text": prompt,
                    "history": st.session_state.messages[:-1][-10:], 
                    "run_baseline_comparison": run_baseline,
                    "enable_cot": enable_cot 
                }
                
                try:
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        ai_reply = data["results"]["project_empathy_output"]
                        baseline_reply = data["results"].get("standard_llama_output")
                        
                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                        st.session_state.latest_metadata = data.get("metadata", {})
                        st.session_state.latest_baseline = baseline_reply
                        
                        st.rerun()
                    else:
                        st.error(f"API Error {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")