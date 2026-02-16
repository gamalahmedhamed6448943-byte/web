import streamlit as st
import os
import sys
import subprocess
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from gtts import gTTS
from moviepy.editor import *
from moviepy.config import change_settings
from moviepy.video.fx.all import crop, resize, fadein
from PIL import Image, ImageFilter
import numpy as np
import textwrap
from rake_nltk import Rake
import nltk
import PIL.Image
import uuid
import random

# هذا الكود يعيد تعريف ANTIALIAS إذا كانت مفقودة ليعمل MoviePy
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- إعدادات النظام وتثبيت NLTK ---
try:
    if os.name == 'posix':
        change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
except:
    pass

@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_resources()

# ==============================================================================
# وظائف مساعدة جديدة (للصوت الطويل والانتقالات)
# ==============================================================================

def generate_long_audio(text, lang='en', output_file='audio.mp3'):
    """دالة لتقسيم النص الطويل وتوليد صوت كامل دون قص."""
    # تقسيم النص إلى جمل لتجنب القطع في منتصف الكلمة
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 3000: # حد آمن لـ gTTS
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
        
    # توليد ملفات صوتية لكل جزء
    temp_files = []
    unique_id = uuid.uuid4().hex
    
    try:
        combined_clips = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            chunk_filename = f"temp_chunk_{unique_id}_{i}.mp3"
            tts = gTTS(text=chunk, lang=lang)
            tts.save(chunk_filename)
            temp_files.append(chunk_filename)
            combined_clips.append(AudioFileClip(chunk_filename))
        
        # دمج المقاطع الصوتية
        if combined_clips:
            final_audio = concatenate_audioclips(combined_clips)
            final_audio.write_audiofile(output_file)
            final_audio.close() # إغلاق لتحرير الموارد
            return True
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return False
    finally:
        # تنظيف ملفات الأجزاء المؤقتة
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
    return False

# ==============================================================================
# 1. كود استخراج الصور
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
                best_candidate = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
                return urljoin(base_url, best_candidate[1])
        except:
            pass
    data_src = img_tag.get('data-src') or img_tag.get('data-original')
    if data_src:
        return urljoin(base_url, data_src)
    src = img_tag.get('src')
    if src:
        return urljoin(base_url, src)
    return None

def check_image_size_is_valid(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code != 200:
            response = requests.get(url, stream=True, timeout=5)
        content_length = response.headers.get('Content-Length')
        if content_length:
            size_kb = int(content_length) / 1024
            if size_kb < 6:
                 return False
        return True
    except:
        return False

def advanced_extract_images(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    extracted_images = []
    try:
        st.info(f"🔄 جاري استخراج الصور من: {url} ...")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript', 'iframe', 'svg']):
            tag.decompose()
        target_area = soup.find('article') or soup.find('main') or soup.find(role='main') or soup.find(id=lambda x: x and 'content' in x)
        if not target_area: target_area = soup
        img_tags = target_area.find_all('img')
        seen_urls = set()

        for i, img in enumerate(img_tags):
            full_url = get_best_image_url(img, url)
            if not full_url: continue
            full_url = full_url.split('?')[0]
            ext_check = full_url.lower()
            if ext_check.endswith('.svg') or ext_check.endswith('.gif') or ext_check.endswith('.ico'): continue
            if 'data:image' in ext_check and len(ext_check) < 1000: continue
            bad_words = ['logo', 'icon', 'avatar', 'profile', 'sprite', 'pixel', 'blank', 'transparent']
            if any(w in ext_check for w in bad_words): continue
            if full_url in seen_urls: continue
            if check_image_size_is_valid(full_url):
                extracted_images.append(full_url)
                seen_urls.add(full_url)

        if not extracted_images:
            st.warning("❌ لم يتم العثور على صور محتوى حقيقية.")
        else:
            st.success(f"🎉 تم استخراج {len(extracted_images)} صورة.")
        return extracted_images
    except Exception as e:
        st.error(f"❌ خطأ في الصور: {e}")
        return []

# ==============================================================================
# 2. كود استخراج النصوص
# ==============================================================================

def extract_text_content_data(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    try:
        st.info(f"🔄 جاري جلب النص من: {url} ...")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article_title = "No Title Found"
        h1 = soup.find('h1')
        if h1:
            article_title = h1.get_text(strip=True)
        else:
            title_tag = soup.find('title')
            if title_tag:
                article_title = title_tag.get_text(strip=True).split('-')[0].strip()

        useless_tags = ['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript', 'iframe', 'svg', 'form', 'button', 'figcaption', 'figure', 'video']
        for tag in list(soup(useless_tags)): tag.decompose()
        
        bad_classes = ['ad', 'advert', 'social', 'share', 'cookie', 'popup', 'promo', 'related-content', 'outbrain', 'taboola']
        for tag in list(soup.find_all(True)):
            if tag is None: continue
            try:
                classes = tag.get('class', [])
                if classes:
                    class_str = " ".join(classes).lower()
                    if any(bad in class_str for bad in bad_classes):
                        tag.decompose()
            except: pass

        target_area = soup.find('article') or soup.find('div', class_=lambda x: x and 'article' in x.lower() and 'body' in x.lower()) or soup.find('div', class_=lambda x: x and 'content' in x.lower()) or soup.find('main')
        if not target_area: target_area = soup
        
        paragraphs = []
        raw_text_list = []
        elements = target_area.find_all(['p', 'h2', 'h3'])
        
        for element in elements:
            text = element.get_text(strip=True)
            if len(text) < 20 and element.name == 'p': continue
            forbidden_phrases = ["Read more", "Follow us", "Copyright", "All rights reserved", "Image source", "Sign up", "Click here", "Ad Feedback", "Story highlights", "CNN", "BBC"]
            if any(phrase.lower() in text.lower() for phrase in forbidden_phrases): continue
            
            raw_text_list.append(text)
            if element.name in ['h2', 'h3']:
                paragraphs.append(f"<h3>{text}</h3>")
            else:
                paragraphs.append(f"<p>{text}</p>")

        full_clean_text = ". ".join(raw_text_list)
        st.success(f"🎉 تم استخراج {len(paragraphs)} فقرة نصية.")
        return article_title, full_clean_text, raw_text_list
    except Exception as e:
        st.error(f"❌ خطأ في النصوص: {e}")
        return None, None, None

# ==============================================================================
# 3. محرك إنتاج الفيديو والبيانات الوصفية
# ==============================================================================

def create_moving_backdrop_clip(img_path, duration, screen_size=(1280, 720)):
    # قراءة الصورة
    pil_img = Image.open(img_path)
    
    # تحويل الخلفية: تغيير الحجم مع فلتر ضبابي
    bg_img = pil_img.resize((screen_size[0], screen_size[1]), Image.LANCZOS)
    bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    # جعل الخلفية تتحرك (Zoom) بسرعة (0.1 بدلاً من 0.02)
    bg_clip = ImageClip(np.array(bg_img)).set_duration(duration)
    bg_clip = bg_clip.resize(lambda t: 1 + 0.15 * t)  # حركة سريعة مستمرة
    bg_clip = bg_clip.set_position(('center', 'center'))
    
    # تحضير الصورة الأمامية
    w, h = pil_img.size
    target_h = int(screen_size[1] * 0.9)
    ratio = w / h
    target_w = int(target_h * ratio)
    
    if target_w > screen_size[0] * 0.9:
        target_w = int(screen_size[0] * 0.9)
        target_h = int(target_w / ratio)
        
    fg_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
    fg_clip = ImageClip(np.array(fg_img)).set_duration(duration)
    
    # إضافة حركة خفيفة للصورة الأمامية أيضاً لتتناغم مع الخلفية
    fg_clip = fg_clip.resize(lambda t: 1 + 0.05 * t)
    fg_clip = fg_clip.set_position(('center', 'center'))
    
    final_clip = CompositeVideoClip([bg_clip, fg_clip], size=screen_size).set_duration(duration)
    return final_clip

def generate_youtube_metadata(title, text_list, url):
    full_text = " ".join(text_list)
    r = Rake()
    r.extract_keywords_from_text(full_text)
    keywords = r.get_ranked_phrases()[:15]
    tags = [k for k in keywords if len(k) < 30]
    tags_str = ", ".join(tags)
    
    summary = "\n\n".join(text_list[:3])
    description = f""" {title} \n\n {summary} \n\n 👇 Read the full article here: {url} \n\n #News #{tags[0].replace(' ','')} #{tags[1].replace(' ','') if len(tags)>1 else 'Video'} """.strip()
    
    thumb_prompt = f"A high-quality YouTube thumbnail image representing '{title}'. Professional news style, high contrast, 4k resolution, featuring elements of {tags[0] if tags else 'news'}."
    
    return tags_str, description, thumb_prompt

def process_pipeline(url_input):
    if not url_input:
        st.warning("❌ الرجاء إدخال رابط.")
        return

    # إنشاء معرف فريد للجلسة الحالية لعدم خلط ملفات المستخدمين
    session_id = uuid.uuid4().hex
    
    # 1. استخراج المحتوى
    title, full_text, text_list = extract_text_content_data(url_input)
    if not title or not full_text:
        st.error("❌ فشل في استخراج النص.")
        return
        
    images_urls = advanced_extract_images(url_input)
    if not images_urls:
        st.warning("⚠️ لم يتم العثور على صور، سيتم استخدام شاشة سوداء مع النص.")

    # 2. توليد الصوت (TTS)
    audio_file = f"generated_audio_{session_id}.mp3"
    with st.spinner("🔊 جاري توليد الصوت..."):
        # استخدام النص الكامل (عنوان + محتوى) بدون قص
        tts_text = f"{title}. {full_text}"
        
        # استدعاء الدالة المساعدة لتقسيم ودمج الصوت
        if not generate_long_audio(tts_text, 'en', audio_file):
             # محاولة احتياطية قصيرة إذا فشل الطويل
             tts = gTTS(text=tts_text[:1000], lang='en')
             tts.save(audio_file)
        
        audio_clip = AudioFileClip(audio_file)
        audio_duration = audio_clip.duration
        st.success(f"✅ تم إنشاء الصوت. المدة: {audio_duration:.2f} ثانية")

    # 3. إعداد الفيديو
    output_filename = f"output_video_{session_id}.mp4"
    
    with st.spinner("🎬 جاري معالجة الصور وإنشاء الفيديو..."):
        if images_urls:
            downloaded_images = []
            for i, img_url in enumerate(images_urls):
                try:
                    img_data = requests.get(img_url).content
                    img_name = f"temp_img_{session_id}_{i}.jpg"
                    with open(img_name, 'wb') as handler:
                        handler.write(img_data)
                    downloaded_images.append(img_name)
                except:
                    continue
            
            if not downloaded_images:
                st.error("❌ فشل تحميل الصور.")
                return

            img_duration = audio_duration / len(downloaded_images)
            clips = []
            
            # اختيار نوع انتقال عشوائي (Transition) لهذا الفيديو بالكامل
            # الخيارات: crossfadein (تلاشي), fadein (ظهور من أسود), None (قص مباشر)
            transition_style = random.choice(['crossfade', 'fadein_black', 'sharp'])
            st.info(f"✨ تم اختيار نمط الانتقال: {transition_style}")

            for img_path in downloaded_images:
                clip = create_moving_backdrop_clip(img_path, img_duration)
                
                # تطبيق الانتقال المختار
                if transition_style == 'crossfade':
                    clip = clip.crossfadein(1.0)
                elif transition_style == 'fadein_black':
                    clip = clip.fadein(1.0)
                # إذا كان sharp لا نضيف تأثير
                
                clips.append(clip)
            
            # دمج الكليبات
            # padding=-1 ضروري لعمل crossfade بشكل صحيح
            pad_val = -1 if transition_style == 'crossfade' else 0
            final_video = concatenate_videoclips(clips, method="compose", padding=pad_val)
            
            # التأكد من مطابقة مدة الفيديو للصوت (قد تختلف قليلاً بسبب الانتقالات)
            if final_video.duration < audio_duration:
                # تمديد آخر إطار إذا لزم الأمر
                pass 
        else:
            color_clip = ColorClip(size=(1280, 720), color=(0,0,0), duration=audio_duration)
            txt_clip = TextClip(title, fontsize=70, color='white', size=(1000, None), method='caption')
            txt_clip = txt_clip.set_position('center').set_duration(audio_duration)
            final_video = CompositeVideoClip([color_clip, txt_clip])

        # 4. دمج الصوت وتصدير الفيديو
        final_video = final_video.set_audio(audio_clip)
        
        st.text("⚙️ جاري تصدير الفيديو (Rendering)... FPS = 1")
        # تم ضبط FPS على 1 كما طُلب
        final_video.write_videofile(output_filename, fps=1, codec="libx264", audio_codec="aac")

        # 5. تنظيف الملفات (باستخدام المعرف الفريد)
        if images_urls:
            for f in downloaded_images:
                try: os.remove(f)
                except: pass
        try: os.remove(audio_file)
        except: pass

        # 6. عرض المخرجات
        st.success("✅ COMPLETED SUCCESSFULLY")
        
        tags, desc, thumb = generate_youtube_metadata(title, text_list, url_input)
        
        st.subheader("📋 YOUTUBE DATA")
        st.text_area("Title", title)
        st.text_area("Description", desc)
        st.text_area("Tags", tags)
        st.info(f"**Thumbnail Prompt:** {thumb}")
        
        st.subheader("🎥 FINAL VIDEO")
        st.video(output_filename)
        
        with open(output_filename, "rb") as file:
            st.download_button(
                label="📁 Download Video",
                data=file,
                file_name=f"generated_video_{session_id}.mp4",
                mime="video/mp4"
            )
            
        # حذف الفيديو النهائي بعد التحميل (اختياري لتوفير المساحة)
        # try: os.remove(output_filename)
        # except: pass

# === واجهة التشغيل الرئيسية ===
st.title("🎬 المولد الشامل: من الرابط إلى الفيديو")
st.markdown("### ألصق رابط المقال أدناه")

url_input_user = st.text_input("URL:", placeholder="https://www.bbc.com/news/...")

if st.button("🚀 إنشاء الفيديو"):
    process_pipeline(url_input_user)