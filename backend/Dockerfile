# Backend Dockerfile for Career Planning Multi-Agent System
# Using Debian Bookworm (stable) for consistent package availability
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
# Note: weasyprint requires Cairo, Pango, and GDK-Pixbuf for PDF generation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy and configure entrypoint script FIRST for optional OpenTelemetry auto-instrumentation
# This ensures it's copied before the rest of the code and line endings are preserved
COPY entrypoint.sh /entrypoint.sh

# Convert line endings to Unix format (LF) and make executable
# This handles cases where git didn't convert line endings properly
RUN sed -i 's/\r$//' /entrypoint.sh && \
    chmod +x /entrypoint.sh && \
    echo "Entrypoint script configured successfully"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p outputs/logs outputs/reports data/career_data data/skills_data data/trends_data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint script to support optional auto-instrumentation
ENTRYPOINT ["/entrypoint.sh"]

# Default command (passed to entrypoint)
CMD ["python", "api_server.py"]