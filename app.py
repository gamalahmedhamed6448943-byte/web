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
from moviepy.video.fx.all import crop, resize
from PIL import Image, ImageFilter
import numpy as np
import textwrap
from rake_nltk import Rake
import nltk
import uuid
import shutil
import random
import PIL.Image

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
# 1. كود استخراج الصور (كما هو)
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
# 2. كود استخراج النصوص (كما هو)
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
# 3. محرك إنتاج الفيديو والبيانات الوصفية (معدل للسرعة والتعددية)
# ==============================================================================

def create_moving_backdrop_clip(img_path, duration, screen_size=(1280, 720), zoom_direction='in', speed_factor=0.06):
    """
    تم التعديل:
    1. zoom_direction: لتغيير اتجاه الحركة عشوائياً.
    2. speed_factor: زادت السرعة من 0.02 إلى 0.06 لتكون 'fast and continuous'.
    """
    pil_img = Image.open(img_path)
    
    # تحضير الخلفية المضببة
    bg_img = pil_img.resize((screen_size[0], screen_size[1]), Image.LANCZOS)
    bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    bg_clip = ImageClip(np.array(bg_img)).set_duration(duration)
    
    # حركة الخلفية (سريعة ومستمرة)
    if zoom_direction == 'in':
        # تكبير سريع
        bg_clip = bg_clip.resize(lambda t: 1 + speed_factor * t)
    else:
        # تصغير سريع (يبدأ مكبراً ويصغر)
        bg_clip = bg_clip.resize(lambda t: (1 + speed_factor * duration) - speed_factor * t)
        
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
    fg_clip = fg_clip.set_position(('center', 'center'))
    
    # حركة طفيفة للصورة الأمامية أيضاً لإضافة ديناميكية
    if zoom_direction == 'in':
         fg_clip = fg_clip.resize(lambda t: 1 + (speed_factor/2) * t)
    else:
         fg_clip = fg_clip.resize(lambda t: 1 + (speed_factor/2) * (duration - t))

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

    # --- عزل الجلسة (Multi-User Support) ---
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(os.getcwd(), f"temp_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    # تحديد مسارات الملفات داخل مجلد الجلسة
    audio_file = os.path.join(session_dir, "generated_audio.mp3")
    output_filename = os.path.join(session_dir, "output_video.mp4")

    # تحديد متغيرات عشوائية للفيديو الحالي (Different Slide Transition every time)
    # نختار عشوائياً سرعة الحركة واتجاهها لهذا الفيديو بالكامل أو لكل شريحة
    base_zoom_speed = random.uniform(0.04, 0.08) # حركة سريعة جداً
    transition_duration = random.uniform(0.5, 1.5) # مدة انتقال متغيرة

    # 1. استخراج المحتوى
    title, full_text, text_list = extract_text_content_data(url_input)
    if not title or not full_text:
        st.error("❌ فشل في استخراج النص.")
        shutil.rmtree(session_dir, ignore_errors=True)
        return
        
    images_urls = advanced_extract_images(url_input)
    if not images_urls:
        st.warning("⚠️ لم يتم العثور على صور، سيتم استخدام شاشة سوداء مع النص.")

    # 2. توليد الصوت (TTS)
    with st.spinner("🔊 جاري توليد الصوت..."):
        tts_text = f"{title}. {full_text}"
        if len(tts_text) > 5000:
            st.info("⚠️ النص طويل جداً، سيتم استخدام أول 5000 حرف للصوت.")
            tts_text = tts_text[:5000]
            
        tts = gTTS(text=tts_text, lang='en')
        tts.save(audio_file)
        
        audio_clip = AudioFileClip(audio_file)
        audio_duration = audio_clip.duration
        st.success(f"✅ تم إنشاء الصوت. المدة: {audio_duration:.2f} ثانية")

    # 3. إعداد الفيديو
    with st.spinner("🎬 جاري معالجة الصور وإنشاء الفيديو (60 FPS)..."):
        if images_urls:
            downloaded_images = []
            for i, img_url in enumerate(images_urls):
                try:
                    img_data = requests.get(img_url).content
                    img_name = os.path.join(session_dir, f"temp_img_{i}.jpg")
                    with open(img_name, 'wb') as handler:
                        handler.write(img_data)
                    downloaded_images.append(img_name)
                except:
                    continue
            
            if not downloaded_images:
                st.error("❌ فشل تحميل الصور.")
                shutil.rmtree(session_dir, ignore_errors=True)
                return

            img_duration = audio_duration / len(downloaded_images)
            clips = []
            
            for i, img_path in enumerate(downloaded_images):
                # عشوائية في الاتجاه لكل شريحة لتغيير النمط
                direction = random.choice(['in', 'out'])
                
                clip = create_moving_backdrop_clip(
                    img_path, 
                    img_duration, 
                    zoom_direction=direction, 
                    speed_factor=base_zoom_speed
                )
                
                # تطبيق انتقال (Transition)
                # Crossfade هو الأكثر دعماً وسرعة في المعالجة الخام
                clip = clip.crossfadein(transition_duration)
                clips.append(clip)
            
            # الدمج باستخدام compose لضمان عمل الانتقالات بشكل صحيح
            final_video = concatenate_videoclips(clips, method="compose", padding=-transition_duration)
        else:
            color_clip = ColorClip(size=(1280, 720), color=(0,0,0), duration=audio_duration)
            txt_clip = TextClip(title, fontsize=70, color='white', size=(1000, None), method='caption')
            txt_clip = txt_clip.set_position('center').set_duration(audio_duration)
            final_video = CompositeVideoClip([color_clip, txt_clip])

        # 4. دمج الصوت وتصدير الفيديو (إعدادات السرعة القصوى)
        final_video = final_video.set_audio(audio_clip)
        
        st.text("⚙️ جاري تصدير الفيديو (Rendering) بأقصى سرعة (Ultrafast, Multi-core, 60FPS)...")
        
        # استخدام كل الأنوية المتاحة
        cpu_count = os.cpu_count() or 2
        
        final_video.write_videofile(
            output_filename, 
            fps=60,                  # مطلوب: 60 إطار
            codec="libx264", 
            audio_codec="aac",
            preset="ultrafast",      # مطلوب: أسرع ضغط
            threads=cpu_count        # مطلوب: استخدام كل الأنوية
        )

        # 5. عرض المخرجات والبيانات
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
                file_name=f"generated_video_{session_id[:8]}.mp4",
                mime="video/mp4"
            )

    # 6. تنظيف ملفات الجلسة (Cleanup)
    # لا نقوم بالحذف فوراً إذا كان المستخدم يحتاج للتحميل، ولكن Streamlit يعيد التشغيل عند التفاعل.
    # الحل الأمثل هنا: ترك الملفات حتى يتم الضغط على التحميل، أو الاعتماد على أن النظام يمسح المجلدات القديمة.
    # في Streamlit البسيط، سنترك التنظيف لنهاية الجلسة أو بداية جديدة، 
    # ولكن لضمان المساحة سنقوم بتنظيف مجلدات الجلسات القديمة (اختيارياً)
    # هنا سنكتفي بعدم الحذف الفوري للمجلد للسماح بالتحميل.
    
    # (ملاحظة: لضمان النظافة، يمكن جدولة حذف المجلد لاحقاً، لكن الكود الحالي يعزل كل مستخدم بـ ID)

# === واجهة التشغيل الرئيسية ===
st.title("🎬 المولد الشامل السريع (60FPS Turbo)")
st.markdown("### ألصق رابط المقال أدناه")

url_input_user = st.text_input("URL:", placeholder="https://www.bbc.com/news/...")

if st.button("🚀 إنشاء الفيديو"):
    process_pipeline(url_input_user)