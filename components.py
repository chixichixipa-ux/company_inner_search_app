import streamlit as st
import utils
import constants as ct

def display_app_title():
    st.markdown(f"## {ct.APP_NAME}")

def display_sidebar():
    with st.sidebar:
        st.markdown("## 利用目的")
        st.session_state.mode = st.radio("", [ct.ANSWER_MODE_1, ct.ANSWER_MODE_2], label_visibility="collapsed")
        st.divider()
        st.markdown("**【「社内文書検索」を選択した場合】**")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)
        st.markdown("**【「社内問い合わせ」を選択した場合】**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)

def display_initial_ai_message():
    with st.chat_message("assistant"):
        st.success("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")
        st.warning("具体的に入力したほうが期待通りの回答を得やすいです。", icon=ct.WARNING_ICON)

def display_conversation_log():
    # prefer `chat_history` (used in `app.py`), fall back to `messages` for backward compatibility
    history = st.session_state.get("chat_history") if st.session_state.get("chat_history") is not None else st.session_state.get("messages", [])
    for idx, message in enumerate(history):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                # support both {'text':..} and {'content':..}
                text = message.get("text") or message.get("content") or ""
                st.markdown(text)
            else:
                content = message["content"]
                # bot content may be under 'text' or 'content'
                content = message.get("content") or message.get("text")

                # Support structured search results returned by utils.process_input
                if isinstance(content, dict) and content.get("type") == "search_results":
                    st.markdown(content.get("message", "検索結果:"))
                    for r in content.get("results", []):
                        icon = utils.get_source_icon(r.get("rel_path", r.get("name")))
                        with st.expander(f'{r["name"]} — {r["rel_path"]}'):
                            st.write(r.get("snippet", "") + "...")
                            key = f"comp_full_{idx}_{r['name']}"
                            if st.button(f"全文を表示: {r['name']}", key=key):
                                full = utils.get_full_document(r["name"])
                                st.code(full, language=None)
                else:
                    # Backward-compatible handling for previous content structure
                    try:
                        if content.get("mode") == ct.ANSWER_MODE_1:
                            if not content.get("no_file_path_flg"):
                                st.markdown(content["main_message"])
                                icon = utils.get_source_icon(content["main_file_path"])
                                if "main_page_number" in content:
                                    st.success(f'{content["main_file_path"]}（ページNo.{content["main_page_number"]+1}）', icon=icon)
                                else:
                                    st.success(content["main_file_path"], icon=icon)
                                if content.get("sub_message"):
                                    st.markdown(content["sub_message"])
                                    for sub in content["sub_choices"]:
                                        icon = utils.get_source_icon(sub["source"])
                                        if "page_number" in sub:
                                            st.info(f'{sub["source"]}（ページNo.{sub["page_number"]+1}）', icon=icon)
                                        else:
                                            st.info(sub["source"], icon=icon)
                            else:
                                st.markdown(content["answer"])
                        else:
                            st.markdown(content["answer"])
                            if content.get("file_info_list"):
                                st.divider()
                                st.markdown(f'##### {content["message"]}')
                                for f in content["file_info_list"]:
                                    st.info(f, icon=utils.get_source_icon(f))
                    except Exception:
                        # Fallback: render raw content
                        st.markdown(str(content))

def display_search_llm_response(llm_response):
    if llm_response.get("context") and llm_response.get("answer") != ct.NO_DOC_MATCH_ANSWER:
        main = llm_response["context"][0]
        main_path = main.metadata["source"]
        st.markdown("入力内容に関する情報は、以下のファイルに含まれている可能性があります。")
        icon = utils.get_source_icon(main_path)
        if "page" in main.metadata:
            p = main.metadata["page"]
            st.success(f"{main_path}（ページNo.{p+1}）", icon=icon)
        else:
            st.success(main_path, icon=icon)

        sub_choices, seen = [], {main_path}
        for doc in llm_response["context"][1:]:
            s = doc.metadata["source"]
            if s in seen: continue
            seen.add(s)
            if "page" in doc.metadata:
                sub_choices.append({"source": s, "page_number": doc.metadata["page"]})
            else:
                sub_choices.append({"source": s})

        if sub_choices:
            st.markdown("その他、ファイルありかの候補を提示します。")
            for sub in sub_choices:
                icon = utils.get_source_icon(sub["source"])
                if "page_number" in sub:
                    st.info(f'{sub["source"]}（ページNo.{sub["page_number"]+1}）', icon=icon)
                else:
                    st.info(sub["source"], icon=icon)

        content = {"mode": ct.ANSWER_MODE_1, "main_message": "入力内容に関する情報は、以下のファイルに含まれている可能性があります。", "main_file_path": main_path}
        if "page" in main.metadata: content["main_page_number"] = main.metadata["page"]
        if sub_choices:
            content["sub_message"] = "その他、ファイルありかの候補を提示します。"
            content["sub_choices"] = sub_choices
    else:
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content = {"mode": ct.ANSWER_MODE_1, "answer": ct.NO_DOC_MATCH_MESSAGE, "no_file_path_flg": True}
    return content

def display_contact_llm_response(llm_response):
    st.markdown(llm_response["answer"])
    file_info_list = []
    if llm_response.get("answer") != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()
        st.markdown("##### 情報源")
        seen = set()
        for doc in llm_response.get("context", []):
            path = doc.metadata["source"]
            if path in seen: continue
            seen.add(path)
            if "page" in doc.metadata:
                info = f"{path}（ページNo.{doc.metadata['page']+1}）"
            else:
                info = path
            st.info(info, icon=utils.get_source_icon(path))
            file_info_list.append(info)
    content = {"mode": ct.ANSWER_MODE_2, "answer": llm_response["answer"]}
    if file_info_list:
        content["message"] = "情報源"
        content["file_info_list"] = file_info_list
    return content