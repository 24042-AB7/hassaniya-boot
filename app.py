import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة والجماليات (High-Contrast UI)
# ================================================================
st.set_page_config(
    page_title="Hassaniya AI Chatbot", 
    page_icon="🇲🇷",
    layout="centered"
)

# إضافة CSS مخصص للتباين العالي والوضوح
st.markdown("""
    <style>
    /* خلفية الصفحة - لون رملي فاتح وواضح */
    .stApp {
        background-color: #FDF5E6;
    }
    
    /* تحسين شكل العناوين */
    h1, h2, h3, p {
        color: #2D241E !important; /* بني غامق جداً للوضوح */
    }

    /* تصميم فقاعة المستخدم - لون بني مع نص أبيض */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessage"] .st-emotion-cache-1c7n2ka) {
        background-color: #8B4513 !important;
        color: white !important;
    }
    
    /* تصميم فقاعة المساعد - لون أبيض مع نص بني غامق */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessage"] .st-emotion-cache-1c7n2ka) {
        /* هذا الجزء يتم التحكم به عبر Streamlit افتراضياً، 
           سنعتمد على الألوان الافتراضية للفقاعات مع تحسين النص */
    }

    /* تخصيص لون النص داخل الفقاعات لضمان الوضوح */
    .stChatMessage p {
        color: #2D241E !important;
        font-size: 1.1rem !important;
    }

    /* تحسين شكل شريط الإدخال */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* تصميم الأزرار السريعة */
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #8B4513;
        background-color: white;
        color: #8B4513;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #8B4513;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# 2. تحميل النموذج (الأساس التقني)
# ================================================================
# استبدل 'your-username/hassaniya-gpt2-model' برابط نموذجك في Hugging Face
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
# 3. منطق التوليد (Inference Logic)
# ================================================================
def generate_response(prompt):
    if tokenizer is None or model is None:
        return "عذراً، حدث خطأ في تحميل النموذج."
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, 
        max_length=60, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7, 
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# ================================================================
# 4. واجهة المستخدم (UI Layout)
# ================================================================

# شريط جانبي احترافي
with st.sidebar:
    st.title("🇲🇷 عن المشروع")
    st.info("""
    **المشروع:** تدريب نموذج لغوي للحسانية.
    **الهدف:** دعم اللغات ذات الموارد المحدودة.
    """)
    st.divider()
    st.write("🛠️ **التقنيات:**")
    st.caption("- Transformers (GPT-2)")
    st.caption("- PyTorch")
    st.caption("- Streamlit")
    
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# العنوان الرئيسي
st.title("💬 مساعد الحسانية الذكي")
st.markdown("<p style='text-align: center;'>اسألني عن التقاليد، الطعام، أو الحياة في الصحراء</p>", unsafe_allow_html=True)

# إنشاء ذاكرة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض تاريخ المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- الميزة الجديدة: الأزرار السريعة (Quick Prompts) ---
st.write("💡 **اقتراحات سريعة:**")
cols = st.columns(3)
quick_prompts = ["كيف حالك؟", "أتاي في المحظرة؟", "الخيمة في الصحراء؟"]

for i, p in enumerate(quick_prompts):
    if cols[i % 3].button(p):
        # إضافة سؤال المستخدم
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"):
            st.markdown(p)
        
        # توليد رد المساعد
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                res = generate_response(p)
                st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.rerun() # تحديث الواجهة فوراً

st.divider()

# إدخال المستخدم
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # 1. عرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. توليد ورد المساعد
    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير..."):
            response = generate_response(prompt)
            st.markdown(response)
    
    # 3. حفظ الرد
    st.session_state.messages.append({"role": "assistant", "content": response})
