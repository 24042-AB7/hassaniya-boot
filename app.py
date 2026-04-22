import streamlit as st
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة
# ================================================================
st.set_page_config(
    page_title="Hassaniya AI Chatbot",
    page_icon="🇲🇷",
    layout="centered"
)

# ================================================================
# 2. CSS احترافي (شكل حديث)
# ================================================================
st.markdown("""
<style>

/* خلفية */
body {
    background-color: #0e1117;
    color: white;
}

/* العنوان */
h1 {
    text-align: center;
    color: #58a6ff;
}

/* الشات */
.stChatMessage {
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 10px;
    font-size: 15px;
    line-height: 1.6;
}

/* المستخدم */
[data-testid="stChatMessage-user"] {
    background-color: #1f6feb;
    color: white;
    border-radius: 15px 15px 0px 15px;
}

/* البوت */
[data-testid="stChatMessage-assistant"] {
    background-color: #262730;
    border-radius: 15px 15px 15px 0px;
}

/* إدخال */
.stChatInputContainer {
    background-color: #0e1117;
    border-top: 1px solid #333;
    padding: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* زر */
button {
    border-radius: 10px !important;
}

/* Hover */
.stChatMessage:hover {
    transform: scale(1.01);
    transition: 0.2s;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# 3. تحميل النموذج
# ================================================================
MODEL_ID = "ABMZD/hassaniya-gpt2-model"

@st.cache_resource
def load_model(model_id):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None, None, None

tokenizer, model, device = load_model(MODEL_ID)

# ================================================================
# 4. توليد الرد
# ================================================================
def generate_response(prompt):
    if tokenizer is None or model is None:
        return "❌ النموذج غير متوفر حالياً"

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# ================================================================
# 5. Sidebar
# ================================================================
with st.sidebar:
    st.markdown("## 🤖 Hassaniya AI")
    st.markdown("---")

    st.markdown("### 📌 عن المشروع")
    st.write("نموذج ذكاء اصطناعي للهجة الحسانية")

    st.markdown("### ⚙️ التقنيات")
    st.write("• GPT-2")
    st.write("• Transformers")
    st.write("• Streamlit")

    st.markdown("---")

    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# ================================================================
# 6. العنوان
# ================================================================
st.markdown("<h1>🇲🇷 Hassaniya AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<center style='color: gray;'>تحدث بالحسانية مع الذكاء الاصطناعي</center>", unsafe_allow_html=True)

# ================================================================
# 7. الذاكرة
# ================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ================================================================
# 8. إدخال المستخدم
# ================================================================
if prompt := st.chat_input("اكتب سؤالك بالحسانية..."):

    # عرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # رد البوت
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("يفكر..."):
            response = generate_response(prompt)

            # تأثير الكتابة
            placeholder = st.empty()
            full_text = ""

            for word in response.split():
                full_text += word + " "
                time.sleep(0.03)
                placeholder.markdown(full_text)

    # حفظ الرد
    st.session_state.messages.append({"role": "assistant", "content": response})

# ================================================================
# 9. Footer
# ================================================================
st.markdown("---")
st.markdown(
    "<center style='color: gray;'>صنع لدعم اللغة الحسانية 🇲🇷</center>",
    unsafe_allow_html=True
)
