import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import base64

# ================================================================
# 1. إعدادات الصفحة وبعث "الروح الموريتانية" في التصميم (UI/UX)
# ================================================================
st.set_page_config(
    page_title="مساعد الحسانية | Hassaniya AI",
    page_icon="🇲🇷",
    layout="centered"
)

# دالة مساعدة لتحميل الصور وتحويلها لـ Base64 لاستخدامها كأيقونات مخصصة
# (يمكنك استبدال الروابط بملفات محليّة إذا أردتِ)
def get_base64_image(url):
    # هذه أيقونات تعبيرية مؤقتة، يفضل تحميل صور حقيقية لـ ملحفة/دراعة/براد
    return url # في الوقت الحالي سنستخدم الروابط مباشرة

# روابط لأيقونات تمثل الهوية الموريتانية
icon_user = "https://cdn-icons-png.flaticon.com/512/11107/11107293.png" # تمثيل مبسط لزي تقليدي
icon_bot = "https://cdn-icons-png.flaticon.com/512/3063/3063822.png" # تمثيل لبراد شاي (أتاي)
icon_sidebar = "https://cdn-icons-png.flaticon.com/512/323/323337.png" # علم موريتانيا

# تحسين المظهر باستخدام CSS متقدم (Hassaniya Stylization)
st.markdown(f"""
    <style>
    /* تحميل خط عربي مناسب ومريح للعين */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="st-"] {{
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }}

    /* تخصيص الخلفية بزخرفة موريتانية خفيفة (اختياري) */
    .main {{
        background-color: #fdfcf5; /* لون رملي فاتح جداً */
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4' viewBox='0 0 4 4'%3E%3Cpath fill='%23d2a02a' fill-opacity='0.05' d='M1 3h1v1H1V3zm2-2h1v1H2V1z'%3E%3C/path%3E%3C/svg%3E");
    }}

    /* تصميم فقاعات الدردشة - روح العمارة التقليدية */
    .stChatMessage {{
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid transparent;
    }}
    
    /* فقاعة المستخدم (الملحفة/الدراعة) */
    [data-testid="stChatMessage"]:nth-child(even) {{
        background-color: #e8f5e9; /* أخضر فاتح جداً */
        border-right: 5px solid #006233; /* الأخضر الموريتاني */
        flex-direction: row-reverse !important;
    }}

    /* فقاعة البوت (الأتاي) */
    [data-testid="stChatMessage"]:nth-child(odd) {{
        background-color: #fffde7; /* أصفر فاتح (لون الأتاي) */
        border-right: 5px solid #d2a02a; /* الذهبي الصحراوي */
    }}

    /* تخصيص شريط الجانب (الخيمة) */
    [data-testid="stSidebar"] {{
        background-color: #006233; /* الأخضر الموريتاني */
        color: white;
        border-left: 2px solid #bd1e1e; /* الأحمر القاني الموريتاني */
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* أزرار مخصصة */
    .stButton>button {{
        background-color: #bd1e1e; /* الأحمر الموريتاني */
        color: white;
        border-radius: 20px;
        border: none;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #d2a02a; /* ذهبي عند التحويم */
        color: black;
    }}

    /* تجميل شريط الإدخال */
    .stChatInputContainer {{
        border-top: 2px solid #d2a02a;
        background-color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# 2. تحميل النموذج (الاحتفاظ بالذاكرة عبر @st.cache_resource)
# ================================================================
# ملاحظة: تأكدي من استبدال هذا برابط نموذجك الفعلي
MODEL_ID = "ABMZD/hassaniya-gpt2-model" # نموذج عربي مؤقت للتجربة

@st.cache_resource
def load_model_and_tokenizer(model_id):
    try:
        st.write("🔧 جاري تجهيز 'الخيمة' الرقمية... (تحميل النموذج)")
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        # GPT-2 تحتاج لتحديد pad_token
        tokenizer.pad_token = tokenizer.eos_token 
        model = GPT2LMHeadModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ مشكلة في الخيمة: لم نستطع تحميل النموذج. تأكد من MODEL_ID. الخطأ: {e}")
        return None, None, None

tokenizer, model, device = load_model_and_tokenizer(MODEL_ID)

# ================================================================
# 3. منطق توليد الإجابات (Inference Logic)
# ================================================================
def generate_response(prompt):
    if tokenizer is None or model is None:
        return "المعذرة، 'الأتاي' مازال ما جهز (النموذج غير محمل)."
    
    # تحضير النص: GPT-2 بحاجة لبداية سياق واضحة
    full_prompt = f"السؤال: {prompt}\nالجواب بالجساني:"
    
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=100, # تحديد طول الرد الجديد
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.7, 
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # استخراج الجواب فقط وتنظيفه
    try:
        response = decoded_output.split("الجواب بالجساني:")[-1].strip()
    except:
        response = decoded_output # في حال فشل التقسيم
        
    return response

# ================================================================
# 4. واجهة المستخدم (Hassaniya Chat Interface)
# ================================================================

# العنوان الجانبي (The Sidebar - "الخيمة")
with st.sidebar:
    st.image(icon_sidebar, width=80) 
    st.markdown("<h2 style='text-align: center;'>خيمة الحسانية 🇲🇷</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 10px; border: 1px solid #d2a02a; border-radius: 10px; background-color: rgba(255,255,255,0.1);'>
    <strong>مرحّب بيكم!</strong><br>
    هذا أول نموذج ذكاء اصطناعي نخلوه يتكلم "كلام البيظان" (الحسانية).<br>
    <br>
    <strong>المطور:</strong><br>
    Oumoukelthoum Sidenna
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.write("🛠️ **التقنيات (عتاد الخيمة):**")
    st.caption("Python, PyTorch, Transformers, Streamlit")
    
    if st.button("🗑️ مسح المحادثة (بدِّل المجلس)"):
        st.session_state.messages = []
        st.rerun()

# العنوان الرئيسي في وسط الصفحة
st.markdown("<h1 style='text-align: center; color: #006233;'>💬 مساعد الحسانية الذكي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #bd1e1e;'>شحالكم؟ اسألوني عن أي شيء يخص موريتانيا وثقافتها</p>", unsafe_allow_html=True)

# إنشاء "ذاكرة" للمحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض تاريخ المحادثة مع الأيقونات المخصصة
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar=icon_user):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=icon_bot):
            st.markdown(message["content"])

# منطقة إدخال المستخدم (Chat Input) بلهجة حسانية
if prompt := st.chat_input("اكتب كلامك بالحسانية هون..."):
    
    # 1. عرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=icon_user):
        st.markdown(prompt)

    # 2. توليد وإظهار رد النموذج
    with st.chat_message("assistant", avatar=icon_bot):
        with st.spinner("نخمم... (لحظة من فضلك)"):
            response = generate_response(prompt)
            # إضافة لمسة "أتاي" في الردود الفارغة
            if not response or len(response) < 2:
                response = "ماهي واضحة لي، عاود الصياغة الله يخليك."
            st.markdown(response)
    
    # 3. حفظ رد النموذج في الذاكرة
    st.session_state.messages.append({"role": "assistant", "content": response})

# تذييل الصفحة (Footer)
st.markdown("---")
st.markdown("<center style='color: #bd1e1e; font-weight: bold;'>صنع بكل حب لدعم الثقافة واللغة الحسانية 🇲🇷</center>", unsafe_allow_html=True)
st.markdown("<center style='color: gray; font-size: 0.8em;'>© 2024 جميع الحقوق محفوظة</center>", unsafe_allow_html=True)
