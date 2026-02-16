import streamlit as st
import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from gtts import gTTS
from moviepy.editor import *
from moviepy.config import change_settings
from PIL import Image, ImageFilter
import numpy as np
import textwrap
from rake_nltk import Rake
import nltk
import PIL.Image
import uuid
import random

# --- إصلاح مشكلة Pillow و MoviePy ---
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- إعدادات النظام ---
try:
    if os.name == 'posix':
        change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
except:
    pass

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_resources()

# ==============================================================================
# 1. معالجة الصوت (حل مشكلة انقطاع النص)
# ==============================================================================

def generate_long_audio(text, lang='en', output_file='audio.mp3'):
    """تقسيم النص الطويل لتجنب حدود gTTS ودمج المقاطع."""
    # تنظيف النص
    text = text.replace('"', '').replace("'", "").strip()
    
    # تقسيم النص إلى جمل بناءً على النقاط لتجنب القص في منتصف الكلمة
    # الحد الأقصى لـ gTTS هو حوالي 5000 حرف، نستخدم 3000 للأمان
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 3000:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
        
    chunk_files = []
    # معرف فريد لهذه العملية الصوتية
    audio_uid = uuid.uuid4().hex
    
    try:
        clips = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            
            chunk_filename = f"temp_tts_{audio_uid}_{i}.mp3"
            tts = gTTS(text=chunk, lang=lang)
            tts.save(chunk_filename)
            chunk_files.append(chunk_filename)
            clips.append(AudioFileClip(chunk_filename))
            
        if clips:
            final_audio = concatenate_audioclips(clips)
            final_audio.write_audiofile(output_file, logger=None)
            final_audio.close()
            # إغلاق الكليبات الفردية
            for clip in clips:
                clip.close()
            return True
            
    except Exception as e:
        st.error(f"Error in audio generation: {e}")
        return False
    finally:
        # تنظيف ملفات الأجزاء
        for f in chunk_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
    return False

# ==============================================================================
# 2. استخراج الصور والنصوص
# ==============================================================================

def get_best_image_url(img_tag, base_url):
    srcset = img_tag.get('srcset') or img_tag.get('data-srcset')
    if srcset:
        try:
            candidates = []
            for entry in srcset.split(','):
                parts = entry.strip().split()
                if len(parts) >= 1:
                    url = parts[0]
                    width = 0
                    if len(parts) > 1 and 'w' in parts[1]:
                        width = int(parts[1].replace('w', ''))
                    candidates.append((width, url))
            if candidates:
                best = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
                return urljoin(base_url, best[1])
        except:
            pass
    src = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-original')
    if src:
        return urljoin(base_url, src)
    return None

def check_image_size(url):
    try:
        h = {'User-Agent': 'Mozilla/5.0'}
        r = requests.head(url, headers=h, timeout=3)
        if r.status_code != 200:
            r = requests.get(url, headers=h, stream=True, timeout=3)
        cl = r.headers.get('Content-Length')
        if cl and int(cl) < 6000: # تجاهل الصور الصغيرة جداً (< 6KB)
            return False
        return True
    except:
        return False

def extract_images(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    images = []
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # تنظيف
        for t in soup(['script', 'style', 'svg', 'footer', 'nav']): t.decompose()
        
        target = soup.find('article') or soup.find('main') or soup
        imgs = target.find_all('img')
        
        seen = set()
        for img in imgs:
            u = get_best_image_url(img, url)
            if not u: continue
            u = u.split('?')[0]
            
            # فلترة الامتدادات والكلمات
            if any(x in u.lower() for x in ['.svg', '.gif', 'logo', 'icon', 'avatar']): continue
            
            if u not in seen and check_image_size(u):
                images.append(u)
                seen.add(u)
                
        return images
    except Exception as e:
        st.error(f"Image Error: {e}")
        return []

def extract_text(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # محاولة إيجاد العنوان
        title = "News Video"
        h1 = soup.find('h1')
        if h1: title = h1.get_text(strip=True)
        else: 
            t = soup.find('title')
            if t: title = t.get_text(strip=True).split('-')[0]

        # تنظيف النص
        for t in soup(['script', 'style', 'nav', 'footer', 'aside', 'form']): t.decompose()
        
        target = soup.find('article') or soup.find('main') or soup
        paragraphs = target.find_all(['p', 'h2'])
        
        text_parts = []
        for p in paragraphs:
            txt = p.get_text(strip=True)
            if len(txt) > 30 and "cookie" not in txt.lower():
                text_parts.append(txt)
                
        full_text = ". ".join(text_parts)
        return title, full_text, text_parts
    except Exception as e:
        st.error(f"Text Error: {e}")
        return None, None, None

# ==============================================================================
# 3. إنشاء الفيديو (Backdrop + Transitions)
# ==============================================================================

def create_styled_clip(img_path, duration, screen_size=(1280, 720)):
    """إنشاء مشهد بخلفية متحركة سريعة."""
    try:
        pil_img = Image.open(img_path).convert('RGB')
        
        # الخلفية الضبابية
        bg_img = pil_img.resize(screen_size, Image.LANCZOS)
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=20))
        
        # جعل الخلفية تتحرك بسرعة (Zoom In)
        # 0.2 تعني زيادة 20% في الحجم خلال المدة (حركة ملحوظة)
        bg_clip = ImageClip(np.array(bg_img)).set_duration(duration)
        bg_clip = bg_clip.resize(lambda t: 1 + 0.2 * t) 
        bg_clip = bg_clip.set_position(('center', 'center'))

        # الصورة الأمامية (Foreground)
        # نحافظ على النسبة
        w, h = pil_img.size
        ratio = w / h
        # نجعل الارتفاع 85% من الشاشة
        new_h = int(screen_size[1] * 0.85)
        new_w = int(new_h * ratio)
        if new_w > screen_size[0]:
            new_w = int(screen_size[0] * 0.9)
            new_h = int(new_w / ratio)
            
        fg_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        fg_clip = ImageClip(np.array(fg_img)).set_duration(duration)
        
        # حركة خفيفة للأمامية أيضاً لتتوافق مع الخلفية
        fg_clip = fg_clip.resize(lambda t: 1 + 0.05 * t).set_position(('center', 'center'))

        return CompositeVideoClip([bg_clip, fg_clip], size=screen_size)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def main_pipeline(url):
    # إنشاء Session ID فريد لهذه العملية تماماً
    session_uuid = uuid.uuid4().hex
    
    st.info("⏳ 1. جاري استخراج المحتوى...")
    title, full_text, text_list = extract_text(url)
    images = extract_images(url)
    
    if not full_text:
        st.error("❌ لم يتم العثور على نص.")
        return

    # 1. توليد الصوت الكامل
    st.info("🔊 2. جاري توليد الصوت (قد يستغرق وقتاً للنصوص الطويلة)...")
    audio_filename = f"audio_{session_uuid}.mp3"
    
    # دمج العنوان مع النص
    tts_content = f"{title}. {full_text}"
    if not generate_long_audio(tts_content, output_file=audio_filename):
        st.error("فشل توليد الصوت.")
        return
    
    audio_clip = AudioFileClip(audio_filename)
    total_duration = audio_clip.duration
    st.success(f"✅ تم إنشاء الصوت: {total_duration:.1f} ثانية")

    # 2. معالجة الصور والفيديو
    st.info("🎬 3. جاري معالجة الفيديو (FPS=1)...")
    video_filename = f"video_{session_uuid}.mp4"
    
    downloaded_imgs = []
    try:
        # تحميل الصور
        if images:
            # نأخذ صوراً تكفي لتغطية الصوت أو نكررها
            # لكن يفضل توزيع الوقت بالتساوي
            clip_duration = total_duration / len(images)
            # إذا كانت المدة قصيرة جدا للصورة، نقلل عدد الصور
            if clip_duration < 3 and len(images) > 1:
                # نحتاج صوراً أقل
                needed = int(total_duration / 3) 
                if needed < 1: needed = 1
                images = images[:needed]
                clip_duration = total_duration / len(images)

            for i, img_url in enumerate(images):
                try:
                    r = requests.get(img_url)
                    fname = f"img_{session_uuid}_{i}.jpg"
                    with open(fname, 'wb') as f:
                        f.write(r.content)
                    downloaded_imgs.append(fname)
                except: pass
        
        final_clip = None
        
        if downloaded_imgs:
            clips = []
            
            # اختيار انتقال عشوائي لهذا الفيديو
            # Crossfade: تداخل
            # FadeIn: ظهور من الأسود
            # None: قطع مباشر (سنستخدم تأثيرات MoviePy)
            transition_type = random.choice(['crossfade', 'fadein', 'rotate_enter'])
            st.write(f"✨ Transition Style: **{transition_type}**")

            for img_path in downloaded_imgs:
                clip = create_styled_clip(img_path, clip_duration)
                if clip:
                    # تطبيق الانتقال
                    if transition_type == 'crossfade':
                        # crossfadein يتطلب أن يكون الكليب لاحقاً للكليب السابق في الترتيب الزمني
                        # لكن في concatenate_videoclips نستخدم padding
                        clip = clip.crossfadein(1.0)
                    elif transition_type == 'fadein':
                        clip = clip.fadein(1.0)
                    
                    clips.append(clip)
            
            if clips:
                # إذا اخترنا crossfade نحتاج padding سلبي
                padding = -1 if transition_type == 'crossfade' else 0
                final_clip = concatenate_videoclips(clips, method="compose", padding=padding)
                
                # ضبط الطول تماماً مع الصوت (قد يزيد أو ينقص قليلاً بسبب الانتقالات)
                if final_clip.duration > total_duration:
                    final_clip = final_clip.subclip(0, total_duration)
                # إذا كان أقصر، نمدد الإطار الأخير (نادر الحدوث مع الحسابات)
        
        # إذا لم ننجح في عمل كليبات أو لا توجد صور، نضع خلفية سوداء ونص
        if not final_clip:
            txt_clip = TextClip(title, fontsize=50, color='white', size=(1000, None), method='caption')
            txt_clip = txt_clip.set_position('center').set_duration(total_duration)
            final_clip = CompositeVideoClip([txt_clip], size=(1280, 720))

        # دمج الصوت
        final_clip = final_clip.set_audio(audio_clip)

        # تصدير الفيديو FPS = 1
        final_clip.write_videofile(
            video_filename, 
            fps=1, 
            codec="libx264", 
            audio_codec="aac",
            preset="ultrafast", # لسرعة التحويل
            logger=None
        )
        
        st.success("🎉 تم الانتهاء!")
        st.video(video_filename)
        
        with open(video_filename, "rb") as f:
            st.download_button("تحميل الفيديو 📥", f, file_name="final_video.mp4")
            
        # عرض البيانات
        tags_str = ",".join(text_list[:5]) # مجرد مثال للكلمات المفتاحية
        st.text_area("Title", title)
        st.text_area("Description", full_text[:500] + "...")
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
        import traceback
        st.text(traceback.format_exc())
        
    finally:
        # تنظيف شامل
        if os.path.exists(audio_filename): os.remove(audio_filename)
        # الصور
        for f in downloaded_imgs:
            if os.path.exists(f): os.remove(f)
        # الفيديو (اختياري، يمكن تركه للتحميل ثم حذفه يدوياً أو بآلية أخرى)
        # if os.path.exists(video_filename): os.remove(video_filename)

# ==============================================================================
# الواجهة
# ==============================================================================

st.title("🎞️ صانع الفيديو الآلي (Multi-User Safe)")
u_input = st.text_input("ضع رابط المقال هنا:")

if st.button("ابـدأ الإنشاء 🚀"):
    if u_input:
        main_pipeline(u_input)
    else:
        st.warning("الرجاء إدخال رابط.")