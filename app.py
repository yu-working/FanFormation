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

#===
def yh_news(yh_url="https://tw.news.yahoo.com/sports/"):
    if not yhnews:
        yhnews=[]
    else:
        return
    r = requests.get(yh_url) #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser
    for i in range(5):
        sel = soup.select(f"#scoreboard-group-2 div div:nth-child(2) ul li:nth-child({i+1}) div div a") #取HTML標中的 <div class="title"></div> 中的<a>標籤存入sel
        for s in sel:
            text = re.sub(r'\d+', '', s.text).replace('--', ' / ', 1).replace('--','')
            yhnews.append(text)
#===

def ptt_search(ptt_url="https://www.ptt.cc/bbs/FAPL/index.html"):
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
ptt_url = "https://www.ptt.cc/bbs/FAPL/index.html"

st.session_state.chat_history = pd.read_csv('user_inputs.csv') if 'user_inputs.csv' in os.listdir() else pd.DataFrame(columns=['Uesr', 'Testbed'])
now = datetime.now().strftime('%Y-%m-%d %H:%M') #colab
st.session_state.ptt_df = pd.read_csv('ptt_news.csv') if 'ptt_news.csv' in os.listdir() else ptt_search()


def RAG(Ncsv='N.csv',Mcsv='M.csv'):
    import ast  # for converting embeddings saved as strings back to arrays
    from scipy import spatial  # for calculating vector similarities for search
    Ncsv_dir = Ncsv
    Mcsv_dir = Mcsv
    # models
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    EMBEDDING_MODEL = "text-embedding-ada-002"
    GPT_MODEL = "gpt-4o"
    
    #get data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    Ndata = pd.read_csv(Ncsv_dir, usecols=['Title', 'Content'])
    Mdata = pd.read_csv(Mcsv_dir)
    # 顯示最後 5 筆資料
    Mdata = Mdata.tail(50)
    return Ndata,Mdata

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
        Ndata,Mdata = RAG()
        res = ""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages =  [
            {  # prompt
                'role':'system',
                'content': f"""
                你是一個經常觀看足球賽事的人，請按照使用者的提問，根據當下時間{now}以及現有資料，使用繁體中文進行簡潔的回覆，不得超過三個段落，每個段落也需簡短。
                當使用者以簡稱詢問時，請使用完整名稱如:英格蘭超級足球聯賽、西班牙甲級足球聯賽、馬德里競技俱樂部、皇家馬德里足球俱樂部等進行回覆，
                並且在全稱後面加上英文原文作為註記。
                請先查看歷史資料庫中{Ndata}、{Mdata}、歷史對話紀錄{his}與網站'PTT'上近期的資料{ptt_df}有無相關資料。
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
    st.title('大語言模型應用於足球運動賽事分析')
    st.subheader("歡迎來到Soccer Testbed！")
    st.write("一個專為足球賽事設計的大型語言模型工具，Soccer Testbed可能會發生錯誤。請查核重要資訊。")
    #NEW
    default_text = "請輸入你的問題..."
    preset_text_1 = "預測日本對上印尼的比分。"
    preset_text_2 = "2016年的西甲冠軍是誰?"
    preset_text_3 = "英超目前最新的賽制?"
    preset_text_4 = "最近世界盃資格賽怎麼樣?"
    preset_text_5 = "請為我組建一支最強的隊伍?"

    if 'user_message' not in st.session_state:
        st.session_state.user_message = default_text
    # user input area
    with st.form(key='user_message_form'):
        user_message = st.text_area("請問我能為你做什麼:", value=st.session_state['user_message'], key='user_message_area')
        submit_button = st.form_submit_button(label='傳送')
    st.write("**快速體驗**")
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
        new_entry = pd.DataFrame([{'User': user_message, 'Testbed': get_answer(user_message,his,ptt_df)}])
        st.session_state.chat_history = pd.concat([st.session_state.chat_history, new_entry], ignore_index=True)
        new_entry.to_csv('user_inputs.csv', mode='a', header=not os.path.exists('user_inputs.csv'), index=False)
        
        st.success('complete!')
        
        # get history
        for index, row in st.session_state.chat_history.iloc[::-1].iterrows():
            user = str(row['User'])
            text = str(row['Testbed'])
            
            #print history
            st.markdown("User : " + user, unsafe_allow_html=True)
            st.markdown("Testbed : "+ text, unsafe_allow_html=True)
            st.markdown("---")

if __name__ == "__main__":
    main()
