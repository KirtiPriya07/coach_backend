import subprocess
import sys
import signal

def main():
    # Start the token server process using uvicorn to run the ASGI app
    server_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "server:app",
        "--host", "0.0.0.0",
        "--port", "5001"
    ])
    
    # Start the LiveKit agent process with the "dev" argument
    agent_proc = subprocess.Popen([sys.executable, "agent.py", "dev"])
    
    def terminate_processes(signum, frame):
        print("Terminating processes...")
        server_proc.terminate()
        agent_proc.terminate()
        sys.exit(0)
    
    # Set up signal handler to clean up child processes on exit (e.g., Ctrl+C)
    signal.signal(signal.SIGINT, terminate_processes)
    signal.signal(signal.SIGTERM, terminate_processes)
    
    # Wait for both processes to complete
    try:
        server_proc.wait()
        agent_proc.wait()
    except KeyboardInterrupt:
        terminate_processes(None, None)

if __name__ == "__main__":
    main()
