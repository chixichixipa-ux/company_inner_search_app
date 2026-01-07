import streamlit as st
import utils
import re
import constants as ct


def _local_css():
    st.markdown(
        """
        <style>
        .chat-box {max-width: 900px; margin: 0 auto;}
        .user {background: #e6f2ff; padding:12px; border-radius:12px; margin:8px 0;}
        .bot {background: #f1f8e9; padding:12px; border-radius:12px; margin:8px 0;}
        .sidebar-note{background:#f3f4f6;border-left:4px solid #3b82f6;padding:10px;border-radius:6px;margin-bottom:8px}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=ct.APP_TITLE, layout="wide")
    _local_css()

    col1, col2 = st.columns([1, 3])

    # Sidebar-like column on left
    with col1:
        st.markdown("### 利用目的")
        purpose = st.radio("", ct.PURPOSES)
        # persist selected mode in session_state for other modules to read
        st.session_state.mode = purpose

        st.markdown("---")
        st.markdown("**社内文書検索を選択した場合**")
        st.markdown(
            "<div class='sidebar-note'>入力内容と関連性が高い社内文書のありかを検索できます。例: プロジェクト手順、規程類</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**社内問い合わせを選択した場合**")
        st.markdown(
            "<div class='sidebar-note'>質問・要望に対して、社内文書の情報をもとに回答を作成します。</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### 入力例")
        st.write("- 社員の在籍証明に関するMTGの議事録を探して")
        st.write("- 勤務手続きについて教えてください")

    # Main content
    with col2:
        st.markdown(f"# {ct.APP_TITLE}")
        st.success(ct.WELCOME_MESSAGE)
        st.warning(ct.TIP_MESSAGE)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for item in st.session_state.chat_history:
            if item["role"] == "user":
                st.markdown(f"<div class='user'><b>あなた:</b> {item['text']}</div>", unsafe_allow_html=True)
            else:
                # bot message may be a plain string or a structured search_results dict
                bot_content = item.get("text")
                if isinstance(bot_content, dict) and bot_content.get("type") == "search_results":
                    st.markdown("<div class='bot'><b>bot:</b></div>", unsafe_allow_html=True)
                    st.markdown(f"**{bot_content.get('message')}**")
                    results = bot_content.get("results", [])
                    if results:
                        # Highlight the top result with styled HTML box
                        top = results[0]
                        box_html = f"""
                        <div style='background:#e6ffed;border-left:6px solid #16a34a;padding:12px;border-radius:8px;margin-bottom:8px'>
                          <div style='font-weight:600'>📄 入力内容に関する情報は、以下のファイルに含まれている可能性があります。</div>
                          <div style='margin-top:8px;color:#065f46'>📁 {top['rel_path']}</div>
                        </div>
                        """
                        st.markdown(box_html, unsafe_allow_html=True)
                        st.markdown(top['snippet'] + "...")

                        # Show other candidates as blue boxes
                        if len(results) > 1:
                            st.markdown("**その他、ファイルの候補:**")
                            for i, r in enumerate(results[1:], start=1):
                                candidate_html = f"""
                                <div style='background:#eef2ff;border-left:6px solid #3b82f6;padding:10px;border-radius:6px;margin:6px 0'>
                                  <div>📄 {r['rel_path']}</div>
                                </div>
                                """
                                st.markdown(candidate_html, unsafe_allow_html=True)

                        # Provide expanders for each result with snippet and full-text button
                        for idx, r in enumerate(results):
                            with st.expander(f"{r['name']} — {r['rel_path']}"):
                                st.write(r['snippet'] + "...")
                                safe = re.sub(r"[^0-9a-zA-Z_]", "_", r['name'])
                                btn_key = f"show_{idx}_{safe}"
                                if st.button(f"全文を表示: {r['name']}", key=btn_key):
                                    full = utils.get_full_document(r['name'])
                                    # render as code block for plain text, but attempt markdown if looks like markdown
                                    if isinstance(full, str) and (full.strip().startswith("#") or "|" in full):
                                        st.markdown(full)
                                    else:
                                        st.code(full, language=None)
                else:
                    st.markdown(f"<div class='bot'><b>bot:</b> {bot_content}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Input area
        st.markdown("---")
        st.text_input("", value="", placeholder="こちらからメッセージを送信してください。", key="user_input")

        def _handle_send():
            user_input_val = st.session_state.get("user_input", "")
            if not user_input_val.strip():
                st.warning("入力が空です。メッセージを入力してください。")
                return
            st.session_state.chat_history.append({"role": "user", "text": user_input_val})
            with st.spinner("処理中..."):
                try:
                    mode = st.session_state.get("mode") or purpose
                    if mode == "社内問い合わせ":
                        resp = utils.get_llm_response(
                            user_input_val,
                            mode,
                            retriever=st.session_state.get("retriever"),
                            chat_history=st.session_state.get("chat_history"),
                        )
                    else:
                        resp = utils.process_input(user_input_val, mode)
                except Exception:
                    resp = "処理中にエラーが発生しました。時間を置いて再度お試しください。"
            st.session_state.chat_history.append({"role": "bot", "text": resp})
            # clear input for next run
            st.session_state["user_input"] = ""

        st.button("送信", on_click=_handle_send)


if __name__ == "__main__":
    main()
