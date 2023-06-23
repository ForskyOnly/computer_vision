FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY best.pt ./best.pt

COPY . .


CMD ["streamlit", "run", "0.0.0.0:80", "main.py"]
