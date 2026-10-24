FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run test suite to verify container build integrity
RUN python run_all_tests.py

EXPOSE 8080

CMD ["python", "unified_dashboard.py"]
