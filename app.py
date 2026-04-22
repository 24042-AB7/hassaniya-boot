import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة والجمالية (Premium UI)
# ================================================================
st.set_page_config(
    page_title="Hassaniya AI - المساعد الذكي", 
    page_icon="🇲🇷",
    layout="centered"
)

# إضافة CSS مخصص لتصميم "صحراوي" عصري
st.markdown("""
    <style>
    /* خلفية الصفحة */
    .stApp {
        background: linear-gradient(180deg, #fdfcfb 0%, #e2d1c3 100%);
    }
    
    /* تصميم فقاعات الدردشة */
    .stChatMessage {
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* تحسين العناوين */
    h1 {
        color: #4a3f35;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans_serif;
        text-align: center;
    }
    
    /* تصميم الأزرار السريعة */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #d2b48c;
        background-color: white;
        color: #4a3f35;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d2b48c;
        color: white;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# 2. تحميل النموذج (الأساس التقني)
# ================================================================
MODEL_ID = "ABMZD/hassaniya-gpt2-model" # استبدله برابط نموذجك

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
# 3. واجهة المستخدم (Sidebar Dashboard)
# ================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6134/6134065.png", width=80)
    st.title("لوحة التحكم ⚙️")
    st.markdown("---")
    
    # عرض إحصائيات النموذج بشكل احترافي
    st.subheader("📊 معلومات النموذج")
    col1, col2 = st.columns(2)
    col1.metric("البارامترات", "9M")
    col2.metric("الحالة", "نشط ✅")
    
    st.write("**اللغة:** الحسانية (Hassaniya)")
    st.write("**النوع:** Small Language Model")
    
    st.markdown("---")
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# ================================================================
# 4. منطق الدردشة والتفاعل
# ================================================================
st.title("🇲🇷 مساعد الحسانية الذكي")
st.markdown("<p style='text-align: center; color: #6b5b4b;'>تحدث مع الذكاء الاصطناعي بلهجة أهل الصحراء</p>", unsafe_allow_html=True)

# إنشاء ذاكرة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض تاريخ المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- الميزة الجديدة: الأزرار السريعة (Quick Prompts) ---
st.write("💡 **جرب هذه الأسئلة:**")
cols = st.columns(3)
prompts = ["كيف حالك؟", "أتاي في المحظرة؟", "الخيمة في الصحراء؟"]

for i, p in enumerate(prompts):
    if cols[i % 3].button(p):
        # محاكاة إرسال السؤال
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"):
            st.markdown(p)
        
        # توليد الرد
        with st.chat_message("assistant"):
            with st.spinner("يفكر..."):
                inputs = tokenizer.encode(p, return_tensors="pt").to(device)
                outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(p, "").strip()
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

st.markdown("---")

# إدخال المستخدم
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("يفكر..."):
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
