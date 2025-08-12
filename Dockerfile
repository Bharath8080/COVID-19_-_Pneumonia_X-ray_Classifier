# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create the uploads directory
RUN mkdir -p uploads

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app_new.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app_new.py"]
