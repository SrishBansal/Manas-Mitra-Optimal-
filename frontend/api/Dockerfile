# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Create a non-root user and set permissions (Required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /home/user/app

# Copy requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a writable cache directory for AI models to prevent Permission Denied errors
RUN mkdir -p /home/user/app/.cache && chmod -R 777 /home/user/app/.cache
ENV HF_HOME=/home/user/app/.cache/huggingface

# Copy the rest of the backend code
COPY --chown=user:user . .

# Expose the specific port Hugging Face requires
EXPOSE 7860

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
