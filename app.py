import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة
# ================================================================
st.set_page_config(
    page_title="مساعد الحسانية الذكي",
    page_icon="🇲🇷",
    layout="wide"
)

# ================================================================
# 2. CSS الاحترافي — لوحة ألوان صحراوية موريتانية
# ================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+Arabic:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>

/* ── المتغيرات ── */
:root {
    --sand:         #C4A882;
    --sand-light:   #EDE0CC;
    --sand-dark:    #8B6B42;
    --ink:          #1A1208;
    --ink-light:    #3D2B0F;
    --gold:         #B8860B;
    --gold-light:   #D4A520;
    --parchment:    #F7F0E3;
}

/* ── إخفاء عناصر Streamlit الافتراضية ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: #EFEAD9; }
[data-testid="stSidebar"] { background: #1A1208 !important; }
[data-testid="stSidebar"] > div:first-child { background: #1A1208 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── الشريط الجانبي ── */
.sidebar-inner {
    padding: 0;
    font-family: 'Noto Sans Arabic', sans-serif;
    direction: rtl;
    height: 100vh;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}

.sidebar-gold-line {
    height: 3px;
    background: linear-gradient(90deg, #B8860B, #C4A882, #B8860B);
    width: 100%;
}

.sidebar-logo {
    text-align: center;
    padding: 24px 20px 18px;
    border-bottom: 0.5px solid rgba(196,168,130,0.15);
}

.flag-circle {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #006233;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px;
    border: 2px solid rgba(196,168,130,0.4);
    font-size: 28px;
}

.sidebar-title {
    font-family: 'Amiri', serif;
    color: #C4A882;
    font-size: 16px;
    font-weight: 700;
    margin: 0;
}

.sidebar-subtitle {
    color: rgba(196,168,130,0.45);
    font-size: 10px;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
}

.sidebar-section {
    padding: 18px 20px 8px;
}

.sidebar-label {
    color: rgba(196,168,130,0.4);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
}

.info-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 11px;
}

.info-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #B8860B;
    margin-top: 7px;
    flex-shrink: 0;
}

.info-text {
    color: rgba(196,168,130,0.75);
    font-size: 12px;
    line-height: 1.6;
}

.info-text strong {
    color: #C4A882;
    font-weight: 500;
}

.tech-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 4px 20px 16px;
}

.tech-pill {
    background: rgba(196,168,130,0.08);
    border: 0.5px solid rgba(196,168,130,0.2);
    color: rgba(196,168,130,0.7);
    font-size: 10px;
    padding: 3px 9px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
}

.sidebar-dunes {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    opacity: 0.06;
    pointer-events: none;
    width: 100%;
}

/* ── المحتوى الرئيسي ── */
.main-wrapper {
    background: #F7F0E3;
    min-height: 100vh;
    font-family: 'Noto Sans Arabic', sans-serif;
    direction: rtl;
    display: flex;
    flex-direction: column;
}

.topbar {
    padding: 16px 28px;
    background: rgba(247,240,227,0.97);
    border-bottom: 0.5px solid rgba(139,107,66,0.2);
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}

.topbar-title {
    font-family: 'Amiri', serif;
    font-size: 22px;
    color: #1A1208;
    font-weight: 700;
    margin: 0;
}

.topbar-caption {
    font-size: 12px;
    color: #8B6B42;
    margin-top: 2px;
}

.status-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    background: rgba(139,107,66,0.08);
    border: 0.5px solid rgba(139,107,66,0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 11px;
    color: #8B6B42;
    font-family: 'IBM Plex Mono', monospace;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #006233;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.35; }
}

/* ── فقاعات المحادثة ── */
.chat-container {
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    flex: 1;
}

.msg-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

.msg-row.user-row { flex-direction: row-reverse; }

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}

.avatar-ai {
    background: #1A1208;
    border: 1.5px solid rgba(184,134,11,0.45);
    color: #B8860B;
    font-family: 'Amiri', serif;
    font-size: 16px;
}

.avatar-user {
    background: rgba(139,107,66,0.12);
    border: 0.5px solid rgba(139,107,66,0.3);
    color: #8B6B42;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
}

.bubble {
    max-width: 68%;
    padding: 11px 16px;
    font-size: 14.5px;
    line-height: 1.75;
    direction: rtl;
    text-align: right;
    word-wrap: break-word;
}

.bubble-ai {
    background: #FFFFFF;
    border: 0.5px solid rgba(139,107,66,0.2);
    color: #1A1208;
    border-radius: 14px 14px 14px 4px;
}

.bubble-user {
    background: #1A1208;
    color: #E8D5B0;
    border-radius: 14px 14px 4px 14px;
}

.bubble-time {
    font-size: 10px;
    color: rgba(139,107,66,0.45);
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0 2px;
    text-align: left;
}

.user-row .bubble-time { text-align: right; }

/* ── مؤشر الكتابة ── */
.typing-bubble {
    background: #fff;
    border: 0.5px solid rgba(139,107,66,0.2);
    border-radius: 14px 14px 14px 4px;
    padding: 13px 17px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.typing-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #B8860B;
    animation: bounce 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%,60%,100% { transform: translateY(0); opacity:0.4; }
    30%          { transform: translateY(-5px); opacity:1; }
}

/* ── اقتراحات ── */
.suggestions-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 28px 12px;
    justify-content: flex-end;
}

.suggestion-chip {
    background: rgba(255,255,255,0.8);
    border: 0.5px solid rgba(139,107,66,0.25);
    color: #8B6B42;
    font-size: 12px;
    padding: 5px 14px;
    border-radius: 20px;
    cursor: pointer;
    font-family: 'Noto Sans Arabic', sans-serif;
    transition: all 0.15s;
    direction: rtl;
}
.suggestion-chip:hover {
    background: #1A1208;
    color: #C4A882;
    border-color: transparent;
}

/* ── خانة الإدخال ── */
.input-area {
    padding: 12px 28px 20px;
    border-top: 0.5px solid rgba(139,107,66,0.2);
    background: rgba(247,240,227,0.98);
    display: flex;
    gap: 10px;
    align-items: flex-end;
    position: sticky;
    bottom: 0;
}

.input-wrap {
    flex: 1;
    background: #FFFFFF;
    border: 0.5px solid rgba(139,107,66,0.3);
    border-radius: 12px;
    display: flex;
    align-items: center;
    padding: 10px 14px;
}

.input-wrap:focus-within {
    border-color: rgba(184,134,11,0.5);
    box-shadow: 0 0 0 3px rgba(184,134,11,0.08);
}

.input-field {
    flex: 1;
    border: none;
    background: transparent;
    font-family: 'Noto Sans Arabic', sans-serif;
    font-size: 14px;
    color: #1A1208;
    outline: none;
    resize: none;
    direction: rtl;
    text-align: right;
    line-height: 1.5;
    min-height: 24px;
}

.input-field::placeholder { color: rgba(139,107,66,0.4); }

.send-btn {
    width: 42px; height: 42px;
    border-radius: 10px;
    background: #1A1208;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.15s;
}

.send-btn:hover  { background: #2D1E0F; transform: scale(1.05); }
.send-btn:active { transform: scale(0.96); }

.send-icon { fill: #B8860B; }

.footer-note {
    text-align: center;
    padding: 8px 0 4px;
    font-size: 10px;
    color: rgba(139,107,66,0.4);
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(247,240,227,0.98);
    padding-bottom: 12px;
}

/* ── إخفاء تام لـ st.chat_input الافتراضي ── */
[data-testid="stChatInput"] { display: none !important; }

/* ── شريط الإدخال المخصص ── */
.custom-input-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #F7F0E3;
    border-top: 1px solid rgba(139,107,66,0.22);
    padding: 14px 32px 18px;
    z-index: 999;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.custom-input-row {
    display: flex;
    align-items: center;
    gap: 10px;
    direction: rtl;
}

/* تعديل حقل النص من Streamlit */
.custom-input-bar .stTextInput {
    flex: 1;
    margin: 0 !important;
}

.custom-input-bar .stTextInput > div {
    padding: 0 !important;
}

.custom-input-bar .stTextInput > div > div {
    border: none !important;
    padding: 0 !important;
}

.custom-input-bar .stTextInput input {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    background: #FFFFFF !important;
    border: 1px solid rgba(139,107,66,0.35) !important;
    border-radius: 14px !important;
    color: #1A1208 !important;
    padding: 13px 18px !important;
    font-size: 14.5px !important;
    height: 50px !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    width: 100% !important;
}

.custom-input-bar .stTextInput input:focus {
    border-color: rgba(184,134,11,0.6) !important;
    box-shadow: 0 0 0 3px rgba(184,134,11,0.1) !important;
}

.custom-input-bar .stTextInput input::placeholder {
    color: rgba(139,107,66,0.45) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 13.5px !important;
}

/* زر الإرسال */
.custom-input-bar .stButton > button {
    height: 50px !important;
    width: 50px !important;
    min-width: 50px !important;
    padding: 0 !important;
    background: #1A1208 !important;
    border: none !important;
    border-radius: 14px !important;
    color: #B8860B !important;
    font-size: 20px !important;
    cursor: pointer !important;
    transition: background 0.15s, transform 0.12s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
}

.custom-input-bar .stButton > button:hover {
    background: #2D1E0F !important;
    transform: scale(1.06) !important;
}

.custom-input-bar .stButton > button:active {
    transform: scale(0.96) !important;
}

/* تذييل داخل شريط الإدخال */
.input-footer-note {
    text-align: center;
    font-size: 10px;
    color: rgba(139,107,66,0.38);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.03em;
    margin-top: 2px;
}

/* مساحة سفلية لتعويض الشريط الثابت */
.bottom-spacer {
    height: 100px;
}

/* زر مسح المحادثة في الشريط الجانبي */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(184,134,11,0.1) !important;
    border: 0.5px solid rgba(184,134,11,0.3) !important;
    border-radius: 8px !important;
    color: #B8860B !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(184,134,11,0.2) !important;
    border-color: rgba(184,134,11,0.5) !important;
    transform: none !important;
}

/* ── رسالة ترحيبية ── */
.welcome-box {
    text-align: center;
    padding: 32px 20px;
    border-bottom: 0.5px solid rgba(139,107,66,0.15);
    margin-bottom: 8px;
}

.welcome-ornament {
    font-family: 'Amiri', serif;
    font-size: 36px;
    color: #B8860B;
    opacity: 0.35;
    display: block;
    margin-bottom: 10px;
}

.welcome-text {
    font-family: 'Amiri', serif;
    font-size: 15px;
    color: #8B6B42;
    line-height: 1.9;
}

.model-tag {
    display: inline-block;
    background: rgba(139,107,66,0.08);
    border: 0.5px solid rgba(139,107,66,0.2);
    color: rgba(139,107,66,0.6);
    font-size: 10px;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# 3. تحميل النموذج
# ================================================================
MODEL_ID = "ABMZD/hassaniya-gpt2-model"

@st.cache_resource
def load_model_and_tokenizer(model_id):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None, None, None

tokenizer, model, device = load_model_and_tokenizer(MODEL_ID)

# ================================================================
# 4. منطق توليد الإجابات
# ================================================================
def generate_response(prompt):
    if tokenizer is None or model is None:
        return "عذراً، حدث خطأ أثناء تحميل النموذج."
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# ================================================================
# 5. الشريط الجانبي الاحترافي
# ================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-inner">
        <div class="sidebar-gold-line"></div>
        <div class="sidebar-logo">
            <div class="flag-circle">🇲🇷</div>
            <p class="sidebar-title">مساعد الحسانية</p>
            <p class="sidebar-subtitle">Hassaniya AI · v1.0</p>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-label">عن المشروع</div>
            <div class="info-row">
                <div class="info-dot"></div>
                <div class="info-text"><strong>النموذج:</strong> GPT-2 مُدرَّب على الحسانية</div>
            </div>
            <div class="info-row">
                <div class="info-dot"></div>
                <div class="info-text"><strong>الهدف:</strong> اللغات محدودة الموارد</div>
            </div>
            <div class="info-row">
                <div class="info-dot"></div>
                <div class="info-text"><strong>المطورة:</strong> Oumoukelthoum Sidenna</div>
            </div>
        </div>
        <div class="sidebar-section" style="padding-bottom:0;">
            <div class="sidebar-label">التقنيات المستخدمة</div>
        </div>
        <div class="tech-pills">
            <span class="tech-pill">Python</span>
            <span class="tech-pill">PyTorch</span>
            <span class="tech-pill">GPT-2</span>
            <span class="tech-pill">Transformers</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">HuggingFace</span>
        </div>
        <svg class="sidebar-dunes" viewBox="0 0 300 120" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 80 Q50 40 100 60 Q150 80 200 50 Q250 20 300 45 L300 120 L0 120Z" fill="#C4A882"/>
            <path d="M0 100 Q75 70 150 85 Q225 100 300 75 L300 120 L0 120Z" fill="#C4A882"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================================================================
# 6. الواجهة الرئيسية
# ================================================================

# الشريط العلوي
st.markdown("""
<div class="topbar">
    <div>
        <p class="topbar-title">💬 مساعد الحسانية الذكي</p>
        <p class="topbar-caption">اسأل عن التقاليد، الطعام، أو الحياة في الصحراء</p>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        متصل
    </div>
</div>
""", unsafe_allow_html=True)

# تهيئة ذاكرة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# رسالة الترحيب
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <span class="welcome-ornament">﷽</span>
        <p class="welcome-text">
            أهلاً وسهلاً — ابدأ محادثتك بالحسانية<br>
            اسألني عن التقاليد والطعام والشعر والحياة في الصحراء
        </p>
        <span class="model-tag">ABMZD/hassaniya-gpt2-model</span>
    </div>
    """, unsafe_allow_html=True)

# عرض المحادثة
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-row user-row">
            <div class="avatar avatar-user">أنت</div>
            <div>
                <div class="bubble bubble-user">{msg["content"]}</div>
                <div class="bubble-time">{msg.get("time", "")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar avatar-ai">ح</div>
            <div>
                <div class="bubble bubble-ai">{msg["content"]}</div>
                <div class="bubble-time">{msg.get("time", "")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# اقتراحات سريعة (تظهر فقط في البداية)
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="suggestions-bar">
        <span class="suggestion-chip">ما هو الثيبودين؟</span>
        <span class="suggestion-chip">أخبرني عن الخيمة</span>
        <span class="suggestion-chip">شعر حساني</span>
        <span class="suggestion-chip">تقاليد الأعراس</span>
    </div>
    """, unsafe_allow_html=True)

# مساحة سفلية لتعويض الشريط الثابت
st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)

# ================================================================
# 7. شريط الإدخال الثابت في الأسفل
# ================================================================
import datetime

st.markdown('<div class="custom-input-bar">', unsafe_allow_html=True)
st.markdown('<div class="custom-input-row">', unsafe_allow_html=True)

col_input, col_btn = st.columns([10, 1])

with col_input:
    user_input = st.text_input(
        label="",
        placeholder="اكتب سؤالك بالحسانية هنا...",
        key="user_input_field",
        label_visibility="collapsed"
    )

with col_btn:
    send_clicked = st.button("⬆", key="send_btn")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="input-footer-note">صنع بكل حب لدعم اللغة الحسانية 🇲🇷 — Oumoukelthoum Sidenna</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# معالجة الإرسال (بالضغط على Enter أو الزر)
prompt = None
if send_clicked and user_input.strip():
    prompt = user_input.strip()
elif user_input.strip() and st.session_state.get("last_input") != user_input.strip():
    # كشف الضغط على Enter عبر تغيير القيمة
    prompt = user_input.strip()

if prompt:
    st.session_state["last_input"] = prompt
    now = datetime.datetime.now().strftime("%H:%M")

    # حفظ وعرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt, "time": now})

    # توليد الرد
    with st.spinner(""):
        response = generate_response(prompt)

    # حفظ رد النموذج
    st.session_state.messages.append({"role": "assistant", "content": response, "time": now})
    st.rerun()
