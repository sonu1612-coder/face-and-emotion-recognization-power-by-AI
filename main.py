import subprocess
import sys
import os
import time
import threading
import webbrowser

import urllib.request
import urllib.error

def open_browser():
    """Wait for Flask to fully boot by polling the server, then open the browser."""
    url = "http://127.0.0.1:5000/"
    is_ready = False
    
    # Poll every 0.1s until the neural network server responds
    for _ in range(100):
        try:
            urllib.request.urlopen(url)
            is_ready = True
            break
        except urllib.error.URLError:
            time.sleep(0.1)
            
    if is_ready:
        print("[Engine] Server active! Opening browser in foreground...")
        if os.name == 'nt':
            # Forces browser to the foreground on Windows
            os.system(f"start {url}")
        else:
            webbrowser.open_new(url)

def start_engine():
    print("=========================================")
    print(" CoreAI - Facial Recognition Engine")
    print("=========================================")
    print("Booting neural network server...")
    print("A web browser window will open shortly.")
    print("If it does not, manually open: http://127.0.0.1:5000")
    print("-----------------------------------------")
    print("Press Ctrl+C here if the web UI 'Power Off' fails.")
    print("-----------------------------------------")

    # Open browser after server starts (only when launched from main.py)
    threading.Thread(target=open_browser, daemon=True).start()

    # Run the Flask app
    try:
        process = subprocess.Popen([sys.executable, "app.py"])
        # Keep main.py alive so terminal stays open
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down engine gracefully...")
        if 'process' in locals():
            process.terminate()

if __name__ == "__main__":
    start_engine()
