import streamlit as st #1.27.0
import pandas as pd
import os #colab
from datetime import datetime
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googlesearch import search
from lxml.html import fromstring
import numpy as np

#===
def yahoo_news(keyword):
    yh_news = []
    yahoo_keyword = keyword + "yahoo"
    for j in search(yahoo_keyword, stop=3, pause=2.0, lang='zh-tw'):
            r = requests.get(j)
            try:
                tree = fromstring(r.content)
            except:
                continue
            title = tree.findtext('.//title')
            if title == '403 Forbidden':
                title = ""
            soup = BeautifulSoup(r.text,"html.parser")
            sel = soup.select(f"div.caas-body")
            #for s in sel:
                #print(s.text)
            yh_news.append([title, sel])
    return yh_news
#===last newest
def last_news():
    last = []
    r = requests.get("https://tsna.com/baseball") #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser
    for i in range(10):
        sel = soup.select(f"#__layout > section > main > div > section > div.max-w-screen-xl.mx-auto > div.news-container > div.news__left > div.news__left__news > a:nth-child({i+1}) > div.news__content > div.main-content > div.news__content__title.news__content__title--pc") #取HTML標中的 <div class="title"></div> 中的<a>標籤存入sel
        for s in sel:
            print(s.text)
            last.append(s.text)
    return last
#===

def ptt_search(ptt_url="https://www.ptt.cc/bbs/Baseball/index.html"):
    r = requests.get(ptt_url) #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser
    sel = soup.select("div.title a") #取HTML標中的 <div class="title"></div> 中的<a>標籤存入sel
    #取得PTT文章標題
    title_list = []
    for s in sel:
        a = s.text
        title_list.append(a)
    #取得對應的日期
    date_list = []
    for i in range(len(sel)):
        n = i+2
        od = soup.select(f"#main-container div.r-list-container.action-bar-margin.bbs-screen div:nth-child({n}) div.meta div.date")
        if od == []:
            date_list.append("-")
        else:
            for d in od:
                date_list.append(d.text)
    #合併df並儲存於ptt_news.csv
    ptt_df = pd.DataFrame(
        {'title': title_list,
        'date': date_list
        })
    ptt_df.to_csv('ptt_news.csv', mode='a', header=not os.path.exists('ptt_news.csv'), index=False)
    return ptt_df

his = None
ptt_url = "https://www.ptt.cc/bbs/Baseball/index.html"

st.session_state.chat_history = pd.read_csv('chat_history.csv') if 'chat_history.csv' in os.listdir() else pd.DataFrame(columns=['Uesr', 'AI'])
now = datetime.now().strftime('%Y-%m-%d %H:%M') #colab
st.session_state.ptt_df = pd.read_csv('ptt_news.csv') if 'ptt_news.csv' in os.listdir() else ptt_search()


def RAG(text_list,user_message):
    # 創建一個空列表來存儲嵌入
    embeddings = []
    # 為每個句子生成嵌入
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    for sentence in text_list:
        response = client.embeddings.create(
            input=sentence,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    # 將文本和嵌入存入資料框
    df = pd.DataFrame({
        "text": text_list,
        "ada_embedding": embeddings
    })
    #將使用者提問嵌入
    user_message_embedding = client.embeddings.create(
            input=user_message,
            model="text-embedding-3-small"
        )
    new_sentence_embedding = np.array(user_message_embedding.data[0].embedding)
    # 歸一化現有的嵌入向量
    embeddings = np.array(df["ada_embedding"].tolist())
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 歸一化新句子的嵌入向量
    new_sentence_embedding_normalized = new_sentence_embedding / np.linalg.norm(new_sentence_embedding)
    # 計算新句子與所有文本的餘弦相似度
    cosine_similarities = np.dot(normalized_embeddings, new_sentence_embedding_normalized)
    # 找到最相似的文本
    max_index = np.argmax(cosine_similarities)
    # 提取最相似的文本
    most_similar_text = df.iloc[max_index]["text"]
    rag_result = most_similar_text
    return rag_result

#連結至LLM
def get_answer(user_message,his,ptt_df,
            model="gpt-4o-mini",  # 語言模型
            temperature=0,  # 回應溫度
            max_tokens=500, # 最大的 token 數
            verbose=True,  # 是否顯示除錯除錯訊息
        ):
        api_key = os.getenv("OPENAI_API_KEY")
        load_dotenv()
        if "OPENAI_API_KEY" in os.environ:
            st.write()
        else:
            st.write("Can't get api key")
        client = OpenAI(api_key=api_key)
        #改用project api key
        if "ptt_df" not in st.session_state:
            st.session_state.ptt_df = ptt_search(ptt_url)
        else:
            st.session_state.ptt_df = ptt_search(ptt_url)

        # 發送請求到 OpenAI API，並獲取回應
        keyword_maker = client.chat.completions.create(
            model="gpt-4o-mini",
            messages =  [
            {  # prompt
                'role':'system',
                'content': "將使用者提問轉換為用於搜尋引擎的關鍵字"
            },
            {
                'role':'user',
                'content': f"{user_message}"
            }
        ],
            stream=True,
        )
        keyword = ""
        # 顯示生成的文本
        for chunk in keyword_maker:
            if chunk.choices[0].delta.content is not None:
                keyword += chunk.choices[0].delta.content
        #st.title(keyword) keyword檢視
        yh_news = yahoo_news(keyword)
        last = last_news()
        text_list = last
        rag_result = RAG(text_list,user_message)
        #--------------------------------------------------------------------------#
        res = ""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages =  [
            {  # prompt
                'role':'system',
                'content': f"""
                你是一個經常觀看棒球賽事的人，請按照使用者的提問，根據當下時間{now}以及現有資料，
                使用繁體中文進行簡潔的回覆，不得超過三個段落，每個段落也需簡短。
                當使用者詢問時，請使用完整名稱如:世界棒球12強賽、中華臺北、大谷翔平等進行回覆，
                並且在全稱後面加上英文原文作為註記。
                請最優先參考{rag_result}有無相關資料
                其次查看歷史對話紀錄{his}、論壇'PTT'上近期的資料{ptt_df}、yahoo新聞網站上{yh_news}。
                當你進行預測時，需有準確的比分以提供使用者參考，並且提及球員表現作為分析依據。
                當你進行知識介紹時，可以簡單介紹後結合新聞介紹
                當你提到球員表現時，需準確的提出相關的比賽。
                當你進行近況分析時，可說明球員狀態。
                當你選擇最強隊員時，除了考慮長期的表現，同時也要提到短期是否有出現傷勢而放棄選擇的球員
                """
            },
            {
                'role':'user',
                'content': f"{user_message}"
            }
        ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                res += chunk.choices[0].delta.content
        return res


def main():
    global new_entry
    st.title('歡迎來到FanFormation！')
    scroll_script = """
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            var element = document.getElementById("scroll-here");
            if (element) {
                element.scrollIntoView({ behavior: "smooth" });
            }
        });
    </script>
    """

    # 插入 JavaScript，確保執行但不影響版面
    st.markdown(scroll_script, unsafe_allow_html=True)
    st.write("FanFormation可能會發生錯誤。請查核重要資訊。如有其他疑惑，可以參考[說明](https://github.com/yu-working/FanFormation.git)")
    #NEW
    if "placeholder" not in st.session_state:
        st.session_state.placeholder = "請輸入您的訊息..."
    preset_text_1 = "預測中華臺北對上日本的比分。"
    preset_text_2 = "2024年世界棒球12強賽的冠軍是誰?"
    preset_text_3 = "世界棒球12強賽最新的賽制?"
    preset_text_4 = "2024年世界棒球12強賽冠軍賽的勝負關鍵?"
    preset_text_5 = "請為我組建一支最強的隊伍?"

    st.write("**快速體驗** : 如果您不清楚怎麼使用，可以點擊下列按鈕")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button('預測比分', key='preset_button_1'):
            st.session_state.user_message = preset_text_1
            st.session_state.placeholder = preset_text_1
    with col2:
        if st.button('歷史紀錄', key='preset_button_2'):
            st.session_state.user_message = preset_text_2
            st.session_state.placeholder = preset_text_2
    with col3:
        if st.button('賽事簡介', key='preset_button_3'):
            st.session_state.user_message = preset_text_3
            st.session_state.placeholder = preset_text_3
    with col4:
        if st.button('近況分析', key='preset_button_4'):
            st.session_state.user_message = preset_text_4
            st.session_state.placeholder = preset_text_4
    with col5:
        if st.button('夢幻隊伍', key='preset_button_5'):
            st.session_state.user_message = preset_text_5
            st.session_state.placeholder = preset_text_5
    
    input = st.chat_input(st.session_state.placeholder)
    #ask        
    if input:
        user_message = input
        if user_message:
            st.write("")
        else:
            user_message = ''
        try:
            his = pd.read_csv('chat_history.csv')
        except:
            his = ""
        try:
            ptt_df = pd.read_csv('ptt_news.csv')
        except:
            ptt_df = None
        new_entry = pd.DataFrame([{'User': user_message, 'AI': get_answer(user_message,his,ptt_df)}])
        st.session_state.chat_history = pd.concat([st.session_state.chat_history, new_entry], ignore_index=True)
        new_entry.to_csv('chat_history.csv', mode='a', header=not os.path.exists('chat_history.csv'), index=False)
        
        #st.success('complete!')
        # 設定背景顏色
        bg_color_user = "#f0f8ff"  # 使用者背景色
        bg_color_ai = "#f9f9f9"    # AI 回應背景色

        # 用 st.markdown() 設定 CSS，讓內容區塊固定高度
        chat_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
            .container {{
            text-align: right; /* 確保父容器對齊 */
            }}
            .fixed-height-box {{
                height: 240px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
            }}
            .user-message {{
                display: inline-block;
                background-color: {bg_color_user};
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 5px;
                white-space: pre-wrap;
            }}
            .ai-message {{
                text-align: left;
                background-color: {bg_color_ai};
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 5px;
                white-space: pre-wrap;
            }}
            </style>
        </head>
        <body>

        <div class="fixed-height-box" id="chat-box">
        """

        # **讀取歷史紀錄**
        for _, row in st.session_state.chat_history.iterrows():
            user = str(row['User']).strip()
            text = str(row['AI']).strip().replace("\n", "<br>")

            chat_html += f"""
                <div class="container"><div class="user-message">{user}</div></div>
                <div class="ai-message">{text}</div>
            """

        chat_html += """
        </div>
        <script>
            var chatBox = document.getElementById("chat-box");
            chatBox.scrollTop = chatBox.scrollHeight;
        </script>
        </body></html>
        """
        
        st.components.v1.html(chat_html, height=270)
if __name__ == "__main__":
    main()
