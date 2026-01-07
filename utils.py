import os
from typing import List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import streamlit as st
import zipfile
import constants as ct

# NOTE: langchain imports are attempted lazily inside functions to avoid
# hard import errors when the package is not installed in the environment.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


############################################################
# ヘルパー：ファイル読み込み / 単純検索
############################################################

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_documents(doc_dir: str = None) -> List[Tuple[str, str]]:
    """Load plain-text documents from a directory.

    If `doc_dir` is not provided, prefer the `date` directory when it exists,
    otherwise fall back to `docs`.

    Returns list of tuples: (filename, content)
    """
    # choose directory: prefer `date` when available
    if not doc_dir:
        if os.path.isdir("date"):
            doc_dir = "date"
        else:
            doc_dir = "docs"

    docs = []
    if not os.path.isdir(doc_dir):
        return docs
    for name in sorted(os.listdir(doc_dir)):
        path = os.path.join(doc_dir, name)
        # plain text / markdown files
        if os.path.isfile(path) and name.lower().endswith((".txt", ".md")):
            docs.append((name, _read_text_file(path)))
        # zip archives in `date/` may contain text/markdown files — read them without extracting
        elif os.path.isfile(path) and name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    for zi in sorted(zf.namelist()):
                        # only include top-level text/md files (skip directories)
                        if zi.endswith("/"):
                            continue
                        if zi.lower().endswith((".txt", ".md")):
                            try:
                                raw = zf.read(zi)
                                text = raw.decode("utf-8", errors="ignore")
                            except Exception:
                                text = ""
                            # present name as zipfile/innerpath for source clarity
                            docs.append((f"{name}/{zi}", text))
            except Exception:
                # ignore malformed zips
                continue
    return docs


def search_documents(query: str, docs: List[Tuple[str, str]], top_k: int = 3) -> List[Tuple[str, str]]:
    """Very simple keyword-based search ranking by occurrence count.

    Replace with embedding-based retrieval for production.
    """
    import re
    query = (query or "").lower().strip()
    if not query:
        return []
    # tokenization: prefer kanji-only sequences for Japanese queries, also include kana and ascii tokens
    kanji_tokens = re.findall(r"[\u4E00-\u9FFF]+", query)
    kana_tokens = re.findall(r"[\u3040-\u309F\u30A0-\u30FF]+", query)
    ascii_tokens = [t for t in re.split(r"\s+", query) if t]
    tokens = list(dict.fromkeys(kanji_tokens + ascii_tokens + kana_tokens))  # unique, prefer kanji

    scores = []
    for name, content in docs:
        text = (content or "").lower()
        # count occurrences for each token (fallback to substring matching)
        count = 0
        for tok in tokens:
            if not tok:
                continue
            try:
                count += text.count(tok)
            except Exception:
                pass
        # also give small boost if query as a whole appears in name
        if query in name.lower():
            count += 2
        if count > 0:
            scores.append((count, name, content))

    # If no scores found, try a relaxed substring search using the longest kanji/ascii token
    if not scores and tokens:
        longest = max(tokens, key=len)
        for name, content in docs:
            text = (content or "").lower()
            if longest in text or longest in name.lower():
                scores.append((1, name, content))

    scores.sort(reverse=True)
    results = [(name, content) for _, name, content in scores[:top_k]]
    return results


############################################################
# フォールバック生成 / OpenAI 呼び出し
############################################################

def _fallback_generate(query: str, matched_docs: List[Tuple[str, str]], purpose: str) -> str:
    if not query:
        return "入力が空です。質問を入力してください。"
    if not matched_docs:
        if purpose == "社内文書検索":
            return "関連する社内文書が見つかりませんでした。キーワードを変えて再検索してください。"
        return "回答を生成できませんでした。詳しく入力してください。"

    summary_lines = []
    for name, content in matched_docs:
        snippet = (content or "")[:300].replace("\n", " ")
        summary_lines.append(f"[{name}] の抜粋: {snippet}...")
    return "\n\n".join(summary_lines)


def call_llm(query: str, context_docs: List[Tuple[str, str]], purpose: str) -> str:
    """Call OpenAI ChatCompletion if API key is set; otherwise use fallback.

    This function avoids raising when key is absent. For production, add error handling,
    request throttling, prompt engineering, and logging.
    """
    if not OPENAI_API_KEY:
        return _fallback_generate(query, context_docs, purpose)

    try:
        import openai
        # Build a simple system prompt and user prompt containing top doc excerpts
        docs_text = "\n\n".join([f"DOCUMENT: {name}\n{content[:1000]}" for name, content in context_docs])
        system = (
            "あなたは社内ドキュメントを参照して、簡潔で正確に回答するアシスタントです。"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"ドキュメント:\n{docs_text}\n\n質問: {query}"},
        ]

        # Support both new (openai>=1.0.0) and legacy clients
        # New client: openai.OpenAI()
        try:
            Client = getattr(openai, "OpenAI", None)
            if Client:
                client = Client(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=getattr(ct, "MODEL", "gpt-3.5-turbo"),
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                )
                # extract content robustly
                try:
                    choice = resp.choices[0]
                    msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                    if msg:
                        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    else:
                        content = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
                    if content:
                        return content.strip()
                except Exception:
                    pass

        except Exception:
            # fall through to legacy call
            pass

        # Legacy interface (older openai versions)
        try:
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=getattr(ct, "MODEL", "gpt-3.5-turbo"),
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return _fallback_generate(query, context_docs, purpose)
    except Exception:
        return _fallback_generate(query, context_docs, purpose)


############################################################
# ここから、ユーザー提示の LangChain ベース実装／補助関数
############################################################


def get_source_icon(source: str) -> str:
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    if isinstance(source, str) and source.startswith("http"):
        return getattr(ct, "LINK_SOURCE_ICON", "🔗")
    return getattr(ct, "DOC_SOURCE_ICON", "📄")


def build_error_message(message: str) -> str:
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    common = getattr(ct, "COMMON_ERROR_MESSAGE", "問題が続く場合は管理者にお問い合わせください。")
    return "\n".join([message, common])


def get_llm_response(chat_message: str, purpose: str = "社内問い合わせ", retriever=None, chat_history: list = None) -> str:
    """Obtain a response from an LLM.

    Behavior:
    - If LangChain's `ChatOpenAI` is available, use it to generate an answer (simple message-based call).
    - Otherwise, fall back to the lightweight `call_llm` path used when LangChain is not installed.

    This function avoids hard failures when LangChain is absent, while enabling richer LLM
    behavior when the package is installed.
    """
    # Build simple matched docs from retriever if provided (used by fallback or for prompt enrichment)
    matched = []
    if retriever is not None:
        try:
            docs = retriever.get_relevant_documents(chat_message)
            for d in docs[:3]:
                source = getattr(d, "metadata", {}).get("source", "document")
                content = getattr(d, "page_content", str(d))
                matched.append((source, content))
        except Exception:
            matched = []

    # Try to use LangChain's ChatOpenAI (several package names/paths exist across versions)
    try:
        try:
            # new canonical import in many langchain versions
            from langchain.chat_models import ChatOpenAI
        except Exception:
            # some environments/packages expose a different module name
            from langchain_openai import ChatOpenAI  # type: ignore

        # construct a lightweight messages sequence
        system = getattr(ct, "SYSTEM_PROMPT_INQUIRY", "あなたは社内ドキュメントを参照して、簡潔で正確に回答するアシスタントです。")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": chat_message},
        ]

        llm = ChatOpenAI(model_name=getattr(ct, "MODEL", "gpt-3.5-turbo"), temperature=getattr(ct, "TEMPERATURE", 0.0))

        # Different langchain versions expose different call styles; handle common ones
        try:
            out = llm(messages)
            if isinstance(out, str):
                return out.strip()
            # try to extract content from common shapes
            gen = getattr(out, "generations", None)
            if gen:
                try:
                    return gen[0][0].text.strip()
                except Exception:
                    pass
            content = getattr(out, "content", None)
            if content:
                return content.strip()
        except Exception:
            # fallback to ChatOpenAI.chat or .generate interfaces
            try:
                resp = llm.chat(messages)  # type: ignore
                # resp may contain choices/messages
                try:
                    return resp.choices[0].message.content.strip()
                except Exception:
                    return str(resp)
            except Exception:
                pass

    except Exception:
        # LangChain not available or call failed — continue to fallback
        pass

    # Final fallback: use existing simple call_llm implementation
    return call_llm(chat_message, matched, purpose)


############################################################
# 既存のトップレベル処理（アプリから呼ばれる）
############################################################

def process_input(text: str, purpose: str = "社内文書検索") -> str:
    """Top-level processing: load docs, search, then call LLM (or fallback).

    Keeps behavior safe when API key is not provided.
    """
    docs = load_documents()
    matched = search_documents(text, docs, top_k=3)
    # If no documents loaded, include a helpful note
    if not docs:
        # create example docs to show functionality
        docs = [
            ("サンプル_社内規程.txt", "こちらはサンプルの社内規程です。申請手順について説明があります。"),
            ("サンプル_FAQ.md", "よくある質問と回答のサンプルです。勤務や手続きに関するQ&Aを収録。")
        ]
        matched = search_documents(text, docs, top_k=3)

    # If the user selected document search, return locations/snippets instead of calling LLM
    if purpose == "社内文書検索":
        if not matched:
            # helpful debug info: show how many docs are loaded and some names
            if docs:
                names = [n for n,_ in docs][:5]
                names_str = ", ".join(names)
                return (
                    f"関連する社内文書が見つかりませんでした。キーワードを変えて再検索してください。\n"
                    f"（読み込まれている文書: {len(docs)} 件。先頭: {names_str}）"
                )
            return "関連する社内文書が見つかりませんでした。キーワードを変えて再検索してください。"

        results = []
        for name, content in matched:
            # If the document originated from a zip, name may be 'zipname/innerpath'
            if "/" in name:
                rel_path = os.path.join("date", name)
            else:
                rel_path = os.path.join("docs", name)
            snippet = (content or "").replace("\n", " ")[:240]
            results.append({"name": name, "rel_path": rel_path, "snippet": snippet})

        return {"type": "search_results", "message": "関連性の高い社内文書の所在（上位結果）:", "results": results}

    # For other purposes, call the LLM (or fallback)
    answer = call_llm(text, matched, purpose)
    return answer


def get_full_document(name: str) -> str:
    """Return full text content for a document name previously returned by load_documents.

    The `name` may be either a plain filename (from `docs/`) or a zip entry like
    'zipfile.zip/inner/path.txt' (as produced when reading `date/` archives).
    """
    if not name:
        return ""
    # zip entry
    if "/" in name:
        zipbase, inner = name.split("/", 1)
        zip_path = os.path.join("date", zipbase)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                raw = zf.read(inner)
                return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    # plain file under docs/
    path = os.path.join("docs", name)
    if os.path.isfile(path):
        return _read_text_file(path)
    return ""
