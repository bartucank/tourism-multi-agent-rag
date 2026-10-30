import argparse
import logging
import re
import sys

from langchain_core.messages import HumanMessage

from agents import create_graph
from config import MAX_INPUT
from rag import Retriever


DISCLAIMER = re.compile(
    r"^\W*(yasal uyar|uyar|danışmanlık not|danismanlik not|önemli not|not:|feragat)",
    re.I,
)


def clean(text, keep_citations):
    paragraphs = [p for p in text.split("\n\n") if not DISCLAIMER.match(p.strip())]
    text = "\n\n".join(paragraphs).strip()
    if keep_citations:
        return text
    tag = r"\[\s*K\d+(?:\s*[,;]\s*K?\d+)*\s*\]"
    text = re.sub(rf"\s*\(?\s*{tag}(?:\s*(?:,|;|ve)?\s*{tag})*(?:\s*\))?", "", text)
    text = re.sub(r"\s*\(\s*[,;\s]*\)", "", text)
    return re.sub(r" +([.,;:])", r"\1", text)


def format_result(result, debug):
    lines = [clean(result["messages"][-1].content, debug)]
    if not debug:
        return lines[0]
    if result["plan"]:
        lines.append("\nCalisan uzmanlar: " + " -> ".join(result["plan"]))
    sources = result.get("sources", [])
    if sources:
        lines.append("\nGetirilen kaynaklar (tamami cevapta kullanilmis olmayabilir):")
        for i, source in enumerate(sources, 1):
            page = f", s. {source['page']}" if source["page"] else ", HTML"
            lines.append(f"[K{i}] {source['title']}{page}\n{source['url']}")
    return "\n".join(lines)


def main():
    cli = argparse.ArgumentParser(description="Turizm multi-agent RAG asistani")
    cli.add_argument("--debug", action="store_true",
                     help="Calisan uzmanlari ve getirilen kaynaklari goster")
    args = cli.parse_args()
    if not args.debug:
        logging.disable(logging.WARNING)
    graph = create_graph(Retriever())
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 15}
    print("Turizm Asistani | Cikis: /exit")
    while True:
        try:
            question = input("\nSiz: ")
        except EOFError:
            break
        if question.strip() == "/exit":
            break
        if not question.strip():
            continue
        if len(question) > MAX_INPUT:
            print(f"Soru {MAX_INPUT} karakterden uzun olamaz.")
            continue
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]}, config=config
        )
        print("\nAsistan:", format_result(result, args.debug))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ConnectionError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        print("Ollama, modeller ve build_index.py adimini kontrol edin.", file=sys.stderr)
        raise SystemExit(1)
