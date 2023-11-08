# CUDA-enabled base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Copy the FastAPI and Streamlit files into the container
COPY fastapi_app.py /app/fastapi_app.py
COPY streamlit_app.py /app/streamlit_app.py
COPY preprocess.py /app/preprocess.py
COPY dataset.csv /app/dataset.csv
# Install Python dependencies for both FastAPI and Streamlit
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install -r /app/requirements.txt
# Expose the FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Start FastAPI in the background and Streamlit in the foreground
CMD ["bash", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload & streamlit run streamlit_app.py"]
