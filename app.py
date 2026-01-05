import streamlit as st
import utils
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
                    for r in bot_content.get("results", []):
                        with st.expander(f"{r['name']} — {r['rel_path']}"):
                            st.write(r['snippet'] + "...")
                            if st.button(f"全文を表示: {r['name']}", key=f"show_{r['name']}"):
                                full = utils.get_full_document(r['name'])
                                st.code(full, language=None)
                else:
                    st.markdown(f"<div class='bot'><b>bot:</b> {bot_content}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Input area
        st.markdown("---")
        user_input = st.text_input("", value="", placeholder="こちらからメッセージを送信してください。", key="user_input")

        send = st.button("送信")
        if send:
            if not user_input.strip():
                st.warning("入力が空です。メッセージを入力してください。")
            else:
                st.session_state.chat_history.append({"role": "user", "text": user_input})
                with st.spinner("処理中..."):
                    # Use different handlers depending on purpose
                    try:
                        if purpose == "社内問い合わせ":
                            # Try to use LangChain-aware helper if available
                            resp = utils.get_llm_response(user_input, purpose, retriever=st.session_state.get("retriever"), chat_history=st.session_state.chat_history)
                        else:
                            resp = utils.process_input(user_input, purpose)
                    except Exception:
                        resp = "処理中にエラーが発生しました。時間を置いて再度お試しください。"
                st.session_state.chat_history.append({"role": "bot", "text": resp})
                # clear input
                st.session_state.user_input = ""


if __name__ == "__main__":
    main()
