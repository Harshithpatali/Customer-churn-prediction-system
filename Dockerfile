FROM python:3.10

WORKDIR /app

COPY requirements_render.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh","-c","uvicorn api.main:app --host 0.0.0.0 --port $PORT"]