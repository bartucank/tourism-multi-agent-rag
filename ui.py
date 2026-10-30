import logging

from langchain_core.messages import HumanMessage
import streamlit as st

from agents import create_graph
from config import MAX_INPUT, MODEL
from main import clean
from rag import Retriever

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Turizm Asistanı", page_icon="🏨", layout="centered",
)
st.html("""
<style>
[data-testid="stAppDeployButton"] { display: none; }
</style>
""")


def show_error(message, exc):
    logger.exception(message)
    st.error(f"{message}: {exc}")
    if st.session_state.debug:
        st.exception(exc)


@st.fragment(run_every=1)
def show_audio(message, index):
    task = message.get("speech_task")
    pending = task is not None and not task.done()
    if "speech_error" in message and st.button(
        "Seslendirmeyi yeniden dene", key=f"speak_{index}", disabled=pending,
    ):
        from tts import request_speech

        message.pop("audio", None)
        message.pop("speech_error", None)
        task = message["speech_task"] = request_speech(message["result"]["answer"])
    if task is not None:
        if not task.done():
            st.info(
                "Ses hazırlanıyor… İlk kullanımda paketler kurulup model indirilir. "
                "Bu sırada sohbet etmeye devam edebilirsiniz."
            )
        else:
            try:
                message["audio"] = task.result()
            except Exception as exc:
                logger.exception("Seslendirme başarısız")
                message["speech_error"] = str(exc)
            finally:
                message.pop("speech_task", None)
    if "speech_error" in message:
        st.error("Seslendirme başarısız; yeniden deneyebilirsiniz: " + message["speech_error"])
    if "audio" in message:
        st.audio(
            message["audio"], format="audio/wav",
            autoplay=message.get("autoplay", False),
        )
        st.caption("Bu ses yapay zekâ ile üretilmiştir.")


def show_answer(message, index):
    result = message["result"]
    st.markdown(clean(result["answer"], keep_citations=st.session_state.debug))
    if st.session_state.debug:
        with st.expander("Uzmanlar ve kaynaklar"):
            st.write("Çalışan uzmanlar: " + (" → ".join(result["plan"]) or "Selamlama"))
            if result["sources"]:
                st.caption("Getirilen kaynaklar (tamamı cevapta kullanılmış olmayabilir)")
                for number, source in enumerate(result["sources"], 1):
                    page = f"s. {source['page']}" if source["page"] else "HTML"
                    st.write(f"[K{number}] {source['title']} · {page}")
                    st.write(source["url"])
    if not st.session_state.tts_enabled:
        return
    show_audio(message, index)


with st.sidebar:
    st.header("Turizm Asistanı")
    st.caption("Otel ve seyahat acenteleri için")
    st.toggle("Debug modu", key="debug", value=False)
    st.toggle("Seslendirme (TTS)", key="tts_enabled", value=False)
    if st.session_state.tts_enabled:
        st.caption("EMA Lightning · Türkçe · Apache-2.0")
        st.caption(
            "Yeni yanıtlar otomatik seslendirilir. İlk sesli yanıtta gerekli paketler kurulur, "
            "model indirilir (yaklaşık 34 MB; paketler ayrıca yer kaplar). "
            "Metin cihazda işlenir."
        )
        st.caption("Kurulum arka planda yapılır. TTS'yi kapatmak başlamış işi iptal etmez.")
    st.divider()
    if st.button("Yeni sohbet", use_container_width=True):
        st.session_state.pop("graph", None)
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Yerel model: {MODEL}")
    st.caption("Konuşmalar yalnızca bu tarayıcı oturumunda tutulur.")

st.title("Turizm Asistanı")
st.caption("Mevzuat ve destekler · Finans · İşletme")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Oteliniz veya acenteniz hakkında bir soru sorun; birlikte değerlendirelim.")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message, index)

question = st.chat_input("Sorunuzu yazın…", max_chars=MAX_INPUT)
if question and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Yanıt hazırlanıyor…"):
                if "graph" not in st.session_state:
                    st.session_state.graph = create_graph(Retriever())
                result = st.session_state.graph.invoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"configurable": {"thread_id": "chat"}, "recursion_limit": 15},
                )
            st.session_state.messages.append({
                "role": "assistant",
                "result": {
                    "answer": result["messages"][-1].content,
                    "plan": result["plan"],
                    "sources": [
                        {key: source[key] for key in ("title", "url", "page")}
                        for source in result["sources"]
                    ],
                },
            })
            if st.session_state.tts_enabled:
                from tts import request_speech

                message = st.session_state.messages[-1]
                message["speech_task"] = request_speech(message["result"]["answer"])
                message["autoplay"] = True
        except Exception as exc:
            show_error("Yanıt üretilemedi. Ollama ve indeksi kontrol edin", exc)
        else:
            st.rerun()
