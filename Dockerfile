FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY best.pt ./best.pt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "--server.enableCORS", "false", "main.py"]
