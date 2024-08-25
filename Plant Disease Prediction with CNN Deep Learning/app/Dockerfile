# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80
EXPOSE 80

# Create .streamlit directory and copy Streamlit configuration files
RUN mkdir -p ~/.streamlit
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

# Set the default command to run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]
