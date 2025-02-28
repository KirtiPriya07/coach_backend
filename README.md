# Coach Backend

This project is a Python-based backend for a LiveKit-powered voice AI interview assistant. It comprises a LiveKit agent (which listens for participants and handles voice interactions) and a token server (which generates JWT tokens for clients to join LiveKit rooms).

**Prerequisites:**  
- Python 3.9+ (tested with Python 3.12.4)  
- Docker (for containerization)  
- A LiveKit server with valid API credentials

---

**Setup & Running:**

1. **Clone & Setup Environment:**  
   - **Clone the Repository:**
     ```bash
     git clone https://github.com/KirtiPriya07/coach_backend.git
     cd coach_backend
     ```
   - **Create and Activate a Virtual Environment:**
     ```bash
     python -m venv env
     # Windows:
     .\env\Scripts\activate
     # macOS/Linux:
     source env/bin/activate
     ```
   - **Install Dependencies:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Configure Environment Variables:**  
     Create a `.env` file in the project root with:
     ```dotenv
     LIVEKIT_API_KEY=your_livekit_api_key
     LIVEKIT_API_SECRET=your_livekit_api_secret
     LIVEKIT_URL=wss://your-livekit-server/agent
     OPENAI_API_KEY=your_key
     DEEPGRAM_API_KEY = your_key
     CARTESIA_API_KEY= your_key
     ```

2. **Running Locally:**  
   - **Run the LiveKit Agent (Development Mode):**
     ```bash
     python agent.py dev
     ```
     This command starts the agent that waits for participants and handles voice interactions.
   - **(Optional) Run the Token Server Separately (Flask):**
     ```bash
     python server.py
     ```
     (Note: The token server runs on port 5001.)

3. **Docker Deployment:**  
   - **Create a Dockerfile** in the project root with the following content:
     ```dockerfile
     FROM python:3.12.4
     ENV PYTHONUNBUFFERED=1
     WORKDIR /app
     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt
     COPY . .
     EXPOSE 5001
     CMD ["python", "agent.py", "dev"]
     ```
   - **Build the Docker Image:**
     ```bash
     docker build -t coach-backend .
     ```
   - **Run the Docker Container:**
     ```bash
     docker run -p 5001:5001 coach-backend
     ```
     This maps port 5001 of the container to port 5001 on your host machine.

---

**Troubleshooting:**  
- Ensure your `.env` file is correctly configured and accessible.  
- Verify that `LIVEKIT_URL` is correct and reachable from your environment.  
- Confirm Docker is installed and running (check with `docker --version`).

**Contributing:**  
Feel free to open issues or submit pull requests with improvements or bug fixes.

**License:**  
This project is licensed under the [MIT License](LICENSE).
