FROM python:3.9

# Ajoutez ces lignes pour installer les packages nécessaires pour OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY best.pt ./best.pt

COPY . .

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "80"]