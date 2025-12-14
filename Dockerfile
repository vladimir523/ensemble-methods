FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

RUN adduser --disabled-password --gecos '' appuser

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ensembles /app/ensembles
COPY ui.py /app/ui.py

ENV PYTHONPATH=/app:$PYTHONPATH

RUN chown -R appuser:appuser /app
USER appuser

CMD ["/bin/sh", "-c", "if [ \"$SERVICE\" = \"backend\" ]; then uvicorn ensembles.backend.app:app --host 0.0.0.0 --port 8000; else streamlit run ui.py --server.address=0.0.0.0 --server.port=8501; fi"]
