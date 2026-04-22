import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================================================================
# 1. إعدادات الصفحة (Streamlit UI)
# ================================================================
st.set_page_config(page_title="Hassaniya AI Chatbot", page_icon="🇲🇷")

st.title("🇲🇷 مساعد الحسانية الذكي")
st.write("تحدث مع النموذج باللغة الحسانية (تجربة نموذج صغير)")

# ================================================================
# 2. تحميل النموذج (استخدم اسم مستخدمك واسم النموذج من Hugging Face)
# ================================================================
# استبدل 'your-username/hassaniya-gpt2-model' بالرابط الصحيح الخاص بك
MODEL_ID = "your-username/hassaniya-gpt2-model" 

@st.cache_resource
def load_model_and_tokenizer(model_id):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None, None

# تحميل النموذج عند تشغيل التطبيق
tokenizer, model = load_model_and_tokenizer(MODEL_ID)

# ================================================================
# 3. دالة التوليد (Inference Function)
# ================================================================
def chat_with_model(prompt, max_length=50):
    if tokenizer is None or model is None:
        return "خطأ: لم يتم تحميل النموذج. تأكد من اسم النموذج في الكود."
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        do_sample=True,         
        top_k=50, 
        top_p=0.95,             
        temperature=0.7,       
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ================================================================
# 4. واجهة المستخدم (UI Layout)
# ================================================================
if tokenizer is not None:
    # صندوق إدخال النص
    user_input = st.text_input("اسأل أي سؤال بالحسانية (مثلاً: كيف حالك؟):")

    if st.button("إرسال"):
        if user_input:
            with st.spinner("النموذج يفكر..."):
                answer = chat_with_model(user_input)
                # تنظيف الإجابة من السؤال الأصلي إذا تكرر
                clean_answer = answer.replace(user_input, "").strip()
                st.success(f"🤖 الإجابة: {clean_answer}")
        else:
            st.warning("من فضلك اكتب شيئاً أولاً!")
else:
    st.error("⚠️ فشل تحميل النموذج. تأكد من صحة MODEL_ID في الكود.")

# تذييل الصفحة
st.markdown("---")
st.caption("مشروع تخرج - تدريب نموذج لغوي للحسانية 🇲🇷")
