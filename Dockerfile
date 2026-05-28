FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY bench/ bench/
COPY bench.py run-missing.py smoke_test.py ./
COPY configs/ configs/
COPY fixtures/ fixtures/
COPY analysis/ analysis/

RUN mkdir -p results

ENTRYPOINT ["python", "bench.py"]
CMD ["--help"]
