import streamlit as st #1.27.0
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
import matplotlib
from matplotlib.font_manager import fontManager
import os #colab
from datetime import datetime #caolab
from langchain_community.chat_models import ChatOpenAI #colab chat
from langchain.schema import AIMessage, HumanMessage #colab chat
import openai #colab chat 0.28fail
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
            print(j) #可忽略
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
            for s in sel:
                print(s.text)
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
    date = soup.select("div.r-list-container.action-bar-margin.bbs-screen div:nth-child(2) div.meta div.date") #取HTML標中的 <div class="title"></div> 中的<a>標籤存入sel
    #get title
    title_list = []
    for s in sel:
    #print(s.text)
        a = s.text
        title_list.append(a)
    #get date
    date_list = []
    for i in range(len(sel)):
        n = i+2
        od = soup.select(f"#main-container div.r-list-container.action-bar-margin.bbs-screen div:nth-child({n}) div.meta div.date")
        if od == []:
            date_list.append("-")
        else:
            for d in od:
                date_list.append(d.text)  # 否則加入處理後的日期
    #make a df
    #合併df
    ptt_df = pd.DataFrame(
        {'title': title_list,
        'date': date_list
        })
    ptt_df.to_csv('ptt_news.csv', mode='a', header=not os.path.exists('ptt_news.csv'), index=False)
    #return
    return ptt_df

his = None
ptt_url = "https://www.ptt.cc/bbs/Baseball/index.html"

st.session_state.chat_history = pd.read_csv('user_inputs.csv') if 'user_inputs.csv' in os.listdir() else pd.DataFrame(columns=['Uesr', 'AI'])
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
                你是一個經常觀看棒球賽事的人，請按照使用者的提問，根據當下時間{now}以及現有資料，使用繁體中文進行簡潔的回覆，不得超過三個段落，每個段落也需簡短。
                當使用者詢問時，請使用完整名稱如:世界棒球12強賽、中華臺北、大谷翔平等進行回覆，
                並且在全稱後面加上英文原文作為註記。
                請最優先參考TSNA新聞網站{rag_result}有無相關資料
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
    st.subheader("一個專為運動設計的大型語言模型工具")
    st.write("FanFormation可能會發生錯誤。請查核重要資訊。")
    #NEW
    default_text = "請輸入你的問題..."
    preset_text_1 = "預測中華臺北對上日本的比分。"
    preset_text_2 = "2024年世界棒球12強賽的冠軍是誰?"
    preset_text_3 = "目前世界棒球12強賽最新的賽制?"
    preset_text_4 = "賴清德在中華臺北取得12強冠軍後有何表示?"
    preset_text_5 = "請為我組建一支最強的隊伍?"

    if 'user_message' not in st.session_state:
        st.session_state.user_message = default_text
    # user input area
    with st.form(key='user_message_form'):
        user_message = st.text_area("請問我能為你做什麼:", value=st.session_state['user_message'], key='user_message_area')
        submit_button = st.form_submit_button(label='傳送')
    st.write("**快速體驗**")
    st.write("如果您不清楚怎使用，可以點擊下列按鈕")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button('預測比分', key='preset_button_1'):
            st.session_state.user_message = preset_text_1
            st.experimental_rerun()
    with col2:
        if st.button('歷史紀錄', key='preset_button_2'):
            st.session_state.user_message = preset_text_2
            st.experimental_rerun()
    with col3:
        if st.button('賽事簡介', key='preset_button_3'):
            st.session_state.user_message = preset_text_3
            st.experimental_rerun()
    with col4:
        if st.button('近況分析', key='preset_button_4'):
            st.session_state.user_message = preset_text_4
            st.experimental_rerun()
    with col5:
        if st.button('夢幻隊伍', key='preset_button_5'):
            st.session_state.user_message = preset_text_5
            st.experimental_rerun()

    #NEW post
    if submit_button:
        if user_message:
            st.write("")
        else:
            user_message = ''
        try:
            his = pd.read_csv('user_inputs.csv')
        except:
            his = ""
        try:
            ptt_df = pd.read_csv('ptt_news.csv')
        except:
            ptt_df = None
        new_entry = pd.DataFrame([{'User': user_message, 'AI': get_answer(user_message,his,ptt_df)}])
        st.session_state.chat_history = pd.concat([st.session_state.chat_history, new_entry], ignore_index=True)
        new_entry.to_csv('user_inputs.csv', mode='a', header=not os.path.exists('user_inputs.csv'), index=False)
        
        st.success('complete!')
        bg_color = "#f0f8ff"
        # get history
        for index, row in st.session_state.chat_history.iloc[::-1].iterrows():
            user = str(row['User'])
            text = str(row['AI'])
            
            #print history
            st.markdown(            
                f"""
                <div style="text-align: right;; background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                    {user} : User  
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            st.markdown(            
                f"""
                <div style="text-align: left;; background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                    FanFormation : {text}  
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")

if __name__ == "__main__":
    main()
