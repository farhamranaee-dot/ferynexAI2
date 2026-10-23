import os
import re
import csv
import requests
import streamlit as st

# ==========================================
# تنظیمات
# ==========================================

APP_NAME = "feryNex AI"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "chat_data.csv")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"

REQUEST_TIMEOUT = 180
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """
تو feryNex AI هستی، یک دستیار فارسی‌زبان.

قوانین مکالمه:
- همیشه فارسی روان و طبیعی صحبت کن، مگر اینکه کاربر زبان دیگری بخواهد.
- جمله‌های کاربر را با دقت بفهم و پاسخ مرتبط بده.
- اگر کاربر سلام کرد، فقط یک سلام طبیعی بکن و حالش را بپرس.
- اگر کاربر پرسید «خوبی؟»، بگو «ممنون، خوبم! تو چطوری؟»
- از جمله‌های شکسته و غلط مثل «شما چطورم؟» استفاده نکن.
- دستور زبان فارسی را رعایت کن.
- پاسخ‌ها را روشن و متناسب با سؤال بنویس.
- اطلاعاتی را که نمی‌دانی، از خودت نساز.
- نام تو feryNex AI است.
""".strip()


# ==========================================
# پردازش متن
# ==========================================

def normalize_text(text):
    text = str(text).strip().lower()

    replacements = {
        "ي": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "\u200c": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[!！?؟.,،؛;]+$", "", text)

    return text.strip()


# ==========================================
# دیتاست آموزشی
# ==========================================

STARTER_DATA = [
    ("سلام", "سلام! من feryNex AI هستم. چطور می‌تونم کمکت کنم؟"),
    ("اسمت چیه؟", "من feryNex AI هستم."),
    ("فری نکس چیه؟", "feryNex یک پروژه نرم‌افزاری برای ارائه ابزارها و خدمات مختلف است."),
]


def create_dataset():
    if not os.path.exists(DATA_FILE):
        with open(
            DATA_FILE,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["user", "assistant"])
            writer.writerows(STARTER_DATA)


def load_dataset():
    create_dataset()
    examples = []

    with open(
        DATA_FILE,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not {
            "user",
            "assistant",
        }.issubset(set(reader.fieldnames)):
            raise ValueError(
                "فایل chat_data.csv باید ستون‌های "
                "user و assistant داشته باشد."
            )

        for row in reader:
            question = (row.get("user") or "").strip()
            answer = (row.get("assistant") or "").strip()

            if question and answer:
                examples.append((question, answer))

    return examples


def save_example(question, answer):
    create_dataset()

    with open(
        DATA_FILE,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            question.strip(),
            answer.strip(),
        ])


# ==========================================
# اتصال به Ollama
# ==========================================

def check_ollama():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )

        if response.ok:
            models = response.json().get("models", [])
            return True, models

        return False, []

    except requests.RequestException:
        return False, []


def generate_ollama_answer(question, history):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # اضافه‌کردن تاریخچه مکالمه
    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content", "").strip()

        if role in ("user", "assistant") and content:
            messages.append({
                "role": role,
                "content": content,
            })

    messages.append({
        "role": "user",
        "content": question,
    })

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
    "temperature": 0.4,
    "top_p": 0.85,
    "num_ctx": 4096,
    "repeat_penalty": 1.1,
},
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("message", {}).get(
            "content",
            "",
        ).strip()

        if not answer:
            return "این بار نتونستم پاسخ مناسبی تولید کنم."

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "به Ollama وصل نمی‌شم. مطمئن شو Ollama "
            "در حال اجراست و آدرس محلی آن در دسترسه."
        )

    except requests.exceptions.Timeout:
        return (
            "تولید پاسخ بیش از حد طول کشید. "
            "لطفاً دوباره امتحان کن."
        )

    except requests.exceptions.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else "نامشخص"
        )

        return (
            f"Ollama خطای HTTP {status_code} برگردوند. "
            "مطمئن شو مدل qwen2.5:3b نصب شده."
        )

    except (requests.RequestException, ValueError, KeyError):
        return "هنگام دریافت پاسخ از Ollama مشکلی پیش اومد."


# ==========================================
# رابط کاربری
# ==========================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="💜",
    layout="centered",
)

st.title("💜 feryNex AI")
st.caption("دستیار هوش مصنوعی با مدل Qwen و Ollama")

# ==========================================
# نوار کناری
# ==========================================

st.sidebar.header("تنظیمات")

st.sidebar.write("**مدل:** `qwen2.5:3b`")
st.sidebar.write("**موتور:** Ollama")

if st.sidebar.button("🔌 بررسی اتصال"):
    connected, models = check_ollama()

    if connected:
        model_names = [
            item.get("name", "")
            for item in models
        ]

        if any(
            name == OLLAMA_MODEL
            or name.startswith(OLLAMA_MODEL + ":")
            for name in model_names
        ):
            st.sidebar.success("اتصال برقرار است و مدل پیدا شد.")
        else:
            st.sidebar.warning(
                "Ollama در دسترس است، اما مدل موردنظر "
                "در فهرست مدل‌ها پیدا نشد."
            )
    else:
        st.sidebar.error("اتصال به Ollama برقرار نشد.")

st.sidebar.divider()

st.sidebar.caption(
    "مدل به‌صورت محلی اجرا می‌شود. "
    "برای پاسخ‌گویی، کامپیوتر باید روشن باشد."
)


# ==========================================
# تاریخچه گفتگو
# ==========================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# ورودی چت
# ==========================================

prompt = st.chat_input("پیامت رو بنویس...")


if prompt:
    # تاریخچه قبل از افزودن پیام جدید
    history = list(st.session_state["messages"])

    st.session_state["messages"].append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("feryNex AI داره فکر می‌کنه..."):
            answer = generate_ollama_answer(
                prompt,
                history,
            )

        st.markdown(answer)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
    })


# ==========================================
# افزودن اطلاعات درباره feryNex
# ==========================================

st.divider()

with st.expander("➕ افزودن نمونه آموزشی"):
    st.caption(
        "این بخش نمونه‌ها را در CSV ذخیره می‌کند. "
        "مدل Qwen با اضافه‌شدن این نمونه‌ها خودکار آموزش نمی‌بیند."
    )

    with st.form("add_training_example"):
        new_question = st.text_input("سؤال")
        new_answer = st.text_area("پاسخ")

        submitted = st.form_submit_button("ذخیره نمونه")

        if submitted:
            if not new_question.strip() or not new_answer.strip():
                st.warning("هر دو قسمت را پر کن.")
            else:
                try:
                    save_example(
                        new_question,
                        new_answer,
                    )

                    st.success(
                        "نمونه در فایل chat_data.csv ذخیره شد."
                    )

                except OSError as error:
                    st.error(
                        f"ذخیره نمونه ناموفق بود: {error}"
                    )


# ==========================================
# مدیریت گفتگو
# ==========================================

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ پاک‌کردن تاریخچه"):
        st.session_state["messages"] = []
        st.rerun()

with col2:
    if st.button("🔄 گفت‌وگوی جدید"):
        st.session_state["messages"] = []
        st.rerun()


# ==========================================
# اطلاعات پایین صفحه
# ==========================================

st.divider()

st.caption(
    "feryNex AI | Powered by Ollama and Qwen"
)