import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة والجماليات (Page Config & Styling)
# ================================================================
st.set_page_config(
    page_title="Hassaniya AI Chatbot", 
    page_icon="🇲🇷",
    layout="centered"
)

# إضافة CSS مخصص لتحسين المظهر (الألوان والخطوط)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# 2. تحميل النموذج (الاحتفاظ بالذاكرة عبر @st.cache_resource)
# ================================================================
# استبدل 'your-username/hassaniya-gpt2-model' برابط نموذجك في Hugging Face
MODEL_ID = "your-username/hassaniya-gpt2-model"

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
# 3. منطق توليد الإجابات (Inference Logic)
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
# 4. واجهة المستخدم (Chat Interface)
# ================================================================

# العنوان الجانبي (Sidebar) ليعطي طابعاً احترافياً للمشروع
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6134/6134065.png", width=100) # أيقونة تعبيرية
    st.title("عن المشروع 🇲🇷")
    st.info("""
    **المشروع:** تدريب نموذج لغوي صغير للحسانية.
    **الهدف:** معالجة اللغات ذات الموارد المحدودة.
    **المطور:** Oumoukelthoum Sidenna
    """)
    st.divider()
    st.write("🛠️ **التقنيات المستخدمة:**")
    st.caption("- Python & PyTorch")
    st.caption("- Transformers (GPT-2)")
    st.caption("- Streamlit & Hugging Face")
    
    if st.button("مسح المحادثة 🗑️"):
        st.session_state.messages = []
        st.rerun()

# العنوان الرئيسي في وسط الصفحة
st.title("💬 مساعد الحسانية الذكي")
st.caption("اسأل عن التقاليد، الطعام، أو الحياة في الصحراء")

# إنشاء "ذاكرة" للمحادثة باستخدام session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض تاريخ المحادثة (Chat History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# منطقة إدخال المستخدم (Chat Input)
if prompt := st.chat_input("اكتب سؤالك بالحسانية هنا..."):
    
    # 1. عرض رسالة المستخدم في الواجهة
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. توليد وإظهار رد النموذج
    with st.chat_message("assistant"):
        with st.spinner("يفكر..."):
            response = generate_response(prompt)
            st.markdown(response)
    
    # 3. حفظ رد النموذج في الذاكرة
    st.session_state.messages.append({"role": "assistant", "content": response})

# تذييل الصفحة
st.markdown("---")
st.markdown("<center style='color: gray;'>صنع بكل حب لدعم اللغة الحسانية 🇲🇷</center>", unsafe_allow_html=True)
