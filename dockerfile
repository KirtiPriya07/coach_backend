# Use an official Python runtime as a parent image.
FROM python:3.12.4

# Set environment variables for Python.
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container.
WORKDIR /app

# Copy requirements file first to leverage Docker cache if dependencies haven’t changed.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that your token server uses (5001).
EXPOSE 5001

# Define the command to run your backend.
CMD ["python", "agent.py", "dev"]
