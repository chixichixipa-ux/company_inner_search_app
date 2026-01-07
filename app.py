import streamlit as st
import utils
import re
import constants as ct


def _local_css():
    st.markdown(
        """
        <style>
        /* overall chat layout */
        .chat-box {max-width: 950px; margin: 0 auto;}
        /* user / bot bubbles */
        .user {background: linear-gradient(135deg,#fef3ff,#ede9fe); padding:12px; border-radius:18px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,0.05);}
        .bot {background: linear-gradient(135deg,#f0fdf4,#ecfeff); padding:12px; border-radius:18px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,0.05);}
        .sidebar-note{background:#fbfdff;border-left:6px solid #93c5fd;padding:12px;border-radius:10px;margin-bottom:12px}

        /* top welcome / tip boxes */
        .top-welcome{background:#f0fdf4;border-left:6px solid #34d399;padding:14px;border-radius:12px;margin-bottom:12px;color:#065f46}
        .top-tip{background:#fffbeb;border-left:6px solid #f59e0b;padding:12px;border-radius:12px;margin-bottom:18px;color:#92400e}

        /* search result boxes */
        .result-top{background:linear-gradient(90deg,#ecfdf5,#f0f9ff);border-left:6px solid #10b981;padding:14px;border-radius:12px;margin-bottom:10px}
        .candidate{background:linear-gradient(90deg,#eef2ff,#f8fafc);border-left:6px solid #60a5fa;padding:12px;border-radius:10px;margin:8px 0}

        /* cute header */
        .app-title {font-weight:800; letter-spacing:-0.02em}
        .app-mascot {font-size:28px; margin-right:8px}

        /* input and button styles */
        .stButton>button, .stButton>div>button {background:linear-gradient(90deg,#f472b6,#f59e0b); color:white; border-radius:14px; padding:8px 14px; border:none}
        .stButton>button:hover {filter:brightness(1.02)}
        .stTextInput>div>div>input {border-radius:12px; padding:12px}

        /* make info boxes softer */
        .stInfo, .stSuccess, .stWarning {border-radius:12px}

        /* small responsive tweaks */
        @media (max-width:900px) {
            .chat-box {padding:0 12px}
        }
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
        # custom styled welcome and tip boxes to match mockup
        st.markdown(f"<div class='top-welcome'>{ct.WELCOME_MESSAGE}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='top-tip'>{ct.TIP_MESSAGE}</div>", unsafe_allow_html=True)

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

        # Input area styled
        st.markdown("---")
        col_input, col_send = st.columns([9,1])
        with col_input:
            st.text_input("", value="", placeholder="こちらからメッセージを送信してください。", key="user_input")
        with col_send:
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
                st.session_state["user_input"] = ""

            st.button("送信", on_click=_handle_send)


if __name__ == "__main__":
    main()
