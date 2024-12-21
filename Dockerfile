# base on python 3.9
FROM python:3.9

# setting workdir
WORKDIR /app

# install Streamlit
RUN pip install streamlit==1.27.0 matplotlib langchain openai langchain-community scipy bs4 python-dotenv google lxml


# copy
COPY . .

#update cache
RUN fc-cache -f -v

# open port 8501
EXPOSE 8501

# run streamlit
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]

