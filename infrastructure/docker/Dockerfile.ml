# Multi-stage build for Python ML services
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY ml/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false qross

# Create app directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/qross/.local

# Copy source code
COPY ml/ ./

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R qross:qross /app

# Set PATH to include user packages
ENV PATH=/home/qross/.local/bin:$PATH

# Switch to non-root user
USER qross

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the ML service
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
