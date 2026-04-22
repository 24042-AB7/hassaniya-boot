import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# إعدادات الصفحة
st.set_page_config(page_title="Hassaniya AI Chatbot", page_icon="🇲🇷")

st.title("🇲🇷 مساعد الحسانية الذكي")
st.write("تحدث مع النموذج باللغة الحسانية (تجربة نموذج صغير)")

# 1. تحميل النموذج من Hugging Face (مرة واحدة فقط عند التشغيل)
@st.cache_resource # هذه الميزة تجعل التطبيق سريعاً جداً
def load_model():
    # استبدل YOUR_USERNAME باسم حسابك في Hugging Face
    model_id = "your-username/hassaniya-gpt2-model" 
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model

with st.spinner("جاري تحميل النموذج الذكي... انتظر قليلاً"):
    tokenizer, model = load_load_model()

# 2. واجهة الدردشة
user_input = st.text_input("اسأل أي سؤال بالحسانية (مثلاً: كيف حالك؟):")

if st.button("إرسال"):
    if user_input:
        with st.spinner("النموذج يفكر..."):
            # عملية التوليد
            inputs = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
            outputs = model.generate(
                inputs, 
                max_length=50, 
                do_sample=True, 
                top_k=50, 
                top_p=0.95, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # عرض الإجابة بشكل جميل
            st.success(f"🤖 الإجابة: {response.replace(user_input, '').strip()}")
    else:
        st.warning("من فضلك اكتب شيئاً أولاً!")