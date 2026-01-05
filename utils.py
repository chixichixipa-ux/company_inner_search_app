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
    query = (query or "").lower().strip()
    if not query:
        return []

    scores = []
    for name, content in docs:
        text = (content or "").lower()
        # count keyword occurrences
        count = sum(text.count(tok) for tok in query.split())
        # small boost for title match
        if query in name.lower():
            count += 2
        if count > 0:
            scores.append((count, name, content))

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

    summary_lines = [f"[{name}] の抜粋: {content[:300].replace('\n', ' ')}..." for name, content in matched_docs]
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
    """
    LLM からの回答取得（LangChain を試行し、なければ既存の call_llm にフォールバック）

    Args:
        chat_message: ユーザー入力テキスト
        purpose: 利用目的（"社内文書検索" またはその他）
        retriever: （任意）retriever オブジェクト
        chat_history: （任意）会話履歴

    Returns:
        生成されたテキスト応答
    """
    # Try to use LangChain if available
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.schema import HumanMessage
        # Some langchain versions provide different chain factories; import lazily
        try:
            from langchain.chains import create_history_aware_retriever, create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
        except Exception:
            # fallback names / older APIs
            create_history_aware_retriever = None
            create_retrieval_chain = None
            create_stuff_documents_chain = None

    except Exception:
        # LangChain not available — build simple context_docs from retriever if provided
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
        return call_llm(chat_message, matched, purpose)

    # If we reach here, LangChain is available (at least partially)
    try:
        llm = ChatOpenAI(model_name=getattr(ct, "MODEL", "gpt-3.5-turbo"), temperature=getattr(ct, "TEMPERATURE", 0.0))

        system_prompt = getattr(ct, "SYSTEM_PROMPT_INQUIRY", "あなたは社内ドキュメントを参照して、簡潔で正確に回答するアシスタントです。")
        question_generator_prompt = ChatPromptTemplate.from_messages([
            ("system", getattr(ct, "SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT", system_prompt)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Select doc/inquiry prompt when purpose or session mode indicates
        if purpose == getattr(ct, "ANSWER_MODE_1", "社内文書検索") or st.session_state.get("mode") == getattr(ct, "ANSWER_MODE_1", "社内文書検索"):
            # use document-search style system prompt if available
            question_answer_prompt = ChatPromptTemplate.from_messages([
                ("system", getattr(ct, "SYSTEM_PROMPT_DOC_SEARCH", system_prompt)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

        # If retriever is provided and helper factories exist, build history-aware retriever
        docs_text = []
        if retriever is not None:
            try:
                if create_history_aware_retriever is not None:
                    history_aware_retriever = create_history_aware_retriever(llm, retriever, question_generator_prompt)
                else:
                    history_aware_retriever = None

                if create_stuff_documents_chain is not None:
                    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
                else:
                    question_answer_chain = None

                if create_retrieval_chain is not None and history_aware_retriever is not None and question_answer_chain is not None:
                    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                else:
                    chain = None

                # gather simple docs text for fallback prompt
                try:
                    docs = retriever.get_relevant_documents(chat_message)
                    for d in docs[:3]:
                        docs_text.append(f"DOCUMENT: {getattr(d, 'metadata', {}).get('source', 'doc')}\n{getattr(d, 'page_content', '')[:1000]}")
                except Exception:
                    docs_text = []

            except Exception:
                history_aware_retriever = None
                chain = None

        else:
            history_aware_retriever = None
            question_answer_chain = None
            chain = None

        # Try to invoke chain if available
        if chain is not None:
            try:
                # different langchain versions expose different call styles
                try:
                    result = chain.invoke({"input": chat_message, "chat_history": chat_history or st.session_state.get("chat_history", [])})
                    # result may contain an 'answer' key or text field
                    if isinstance(result, dict):
                        answer = result.get("answer") or result.get("text") or str(result)
                    else:
                        answer = str(result)
                    # append to chat history if possible
                    try:
                        st.session_state.chat_history.extend([HumanMessage(content=chat_message), HumanMessage(content=answer)])
                    except Exception:
                        pass
                    return answer
                except Exception:
                    # fallback to calling as a callable
                    out = chain({"input": chat_message, "chat_history": chat_history or st.session_state.get("chat_history", [])})
                    if isinstance(out, dict):
                        answer = out.get("answer") or out.get("text") or str(out)
                    else:
                        answer = str(out)
                    try:
                        st.session_state.chat_history.extend([HumanMessage(content=chat_message), HumanMessage(content=answer)])
                    except Exception:
                        pass
                    return answer
            except Exception:
                pass

        # If no chain, build a simple prompt including docs_text and call LLM directly
        prompt = getattr(ct, "SYSTEM_PROMPT_INQUIRY", system_prompt) + "\n\n"
        if docs_text:
            prompt += "\n\n".join(docs_text) + f"\n\n質問: {chat_message}"
        else:
            prompt += f"質問: {chat_message}"

        # Try to use ChatOpenAI client
        try:
            # some versions support __call__ with messages
            messages = [
                {"role": "system", "content": getattr(ct, "SYSTEM_PROMPT_INQUIRY", system_prompt)},
                {"role": "user", "content": chat_message},
            ]
            out = llm(messages)
            if isinstance(out, str):
                return out.strip()
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
            pass

    except Exception:
        pass

    # As a last resort, build simple matched docs and call existing call_llm
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
