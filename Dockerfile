FROM python:3.10-slim

# Install system dependencies (including libGL for mediapipe + OpenCV)
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Railway
EXPOSE 8080

# Start app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "api:app"]
