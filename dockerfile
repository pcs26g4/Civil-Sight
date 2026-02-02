FROM python:3.11-slim

WORKDIR /app/Backend

COPY Backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY Backend .
#COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]