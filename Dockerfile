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
RUN pip install Pillow

# Copy all project files
COPY . .

# Create folders for storing data
RUN mkdir -p chiller_detections icons

# Expose the port your app runs on
EXPOSE 8000

# Run icon generation at runtime (not during build)
CMD ["sh", "-c", "python generate_icons.py && python chiller.py"]