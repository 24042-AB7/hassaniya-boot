import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة والتصميم المتقدم (UI/UX Customization)
# ================================================================
st.set_page_config(
    page_title="Hassaniya AI | مساعد الحسانية", 
    page_icon="🇲🇷",
    layout="centered"
)

# تصميم CSS متطور لدعم العربية وجمالية الواجهة
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        flex-direction: row-reverse !important; /* عكس اتجاه فقاعات الدردشة */
        text-align: right !important;
    }
    .main {
        background-color: #f0f2f6;
    }
    /* تلوين شريط الجانب */
    [data-testid="stSidebar"] {
        background-color: #006233; /* الأخضر الموريتاني */
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #d2a02a;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# 2. تحميل النموذج مع معالجة ذكية للـ Tokens
# ================================================================
MODEL_ID = "ABMZD/hassaniya-gpt2-model"

@st.cache_resource
def load_model_and_tokenizer(model_id):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        # GPT-2 ليس لديه pad_token افتراضي، نستخدم eos_token بدلاً منه
        tokenizer.pad_token = tokenizer.eos_token 
        
        model = GPT2LMHeadModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ خطأ: لم يتم العثور على النموذج. تأكد من MODEL_ID: {e}")
        return None, None, None

tokenizer, model, device = load_model_and_tokenizer(MODEL_ID)

# ================================================================
# 3. منطق التوليد مع سياق المحادثة (Context-Aware Inference)
# ================================================================
def generate_response(prompt, history):
    if tokenizer is None or model is None:
        return "المعذرة، النموذج غير جاهز حالياً."

    # بناء السياق: نأخذ آخر رسالتين ليفهم النموذج مجرى الحديث
    context = ""
    for msg in history[-2:]:  
        context += f"User: {msg['content']}\nAssistant: "
    
    full_prompt = context + prompt + "\nAssistant:"
    
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=150, # توليد نص جديد بطول مناسب
            do_sample=True, 
            top_k=40, 
            top_p=0.92, 
            temperature=0.8,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # تنظيف الرد لاستخراج إجابة المساعد فقط
    clean_response = response.split("Assistant:")[-1].strip()
    return clean_response

# ================================================================
# 4. واجهة المستخدم (Interface)
# ================================================================

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🇲🇷</h1>", unsafe_allow_html=True)
    st.title("مساعد الحسانية الذكي")
    st.write("أول نموذج ذكاء اصطناعي متخصص في اللهجة الحسانية الموريتانية.")
    
    st.divider()
    
    # إحصائيات سريعة تعطي طابعاً احترافياً
    col1, col2 = st.columns(2)
    col1.metric("اللغة", "الحسانية")
    col2.metric("الإصدار", "V1.0")
    
    if st.button("🗑️ مسح الذاكرة"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Area
st.markdown("## مرحباً بك في فضاء الحسانية ✨")
st.info("جرب تسألني: 'كيف حالك؟' أو 'أحكي لي عن تقاليد العرس'")

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# إدخال المستخدم
if user_input := st.chat_input("تكلم معاي بالحسانية..."):
    
    # عرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # توليد الرد
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("نخمم..."): # كلمة حسانية بمعنى (أفكر)
            ai_response = generate_response(user_input, st.session_state.messages[:-1])
            if not ai_response: 
                ai_response = "ماني فاهمك زين، حاول توضح لي."
            st.markdown(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# تذييل احترافي
st.markdown("---")
st.caption("تطوير: **Oumoukelthoum Sidenna** | جميع الحقوق محفوظة لتعزيز المحتوى الموريتاني 2024")
