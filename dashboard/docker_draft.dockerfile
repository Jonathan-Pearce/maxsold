FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ ./dashboard/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/maxsold_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]