FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create folders for storing data
RUN mkdir -p chiller_detections icons

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash chiller
RUN chown -R chiller:chiller /app
USER chiller

# Expose the port your app runs on
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with Gunicorn for production
CMD ["sh", "-c", "python generate_icons.py && gunicorn --config gunicorn_config.py wsgi:application"]
