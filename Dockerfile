FROM python:3.9-slim

# Çalışma dizini
WORKDIR /app

# Dosyaları kopyala
COPY app.py /app/
COPY requirements.txt /app/

# Gerekli sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pip güncelle ve paketleri kur
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Render dış dünyaya 8080'den erişir
EXPOSE 8080

# Uygulamayı başlat (Flask yerine gunicorn)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
