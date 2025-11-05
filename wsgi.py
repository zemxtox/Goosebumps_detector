#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn deployment
"""
import os
import threading
from chiller import app, socketio, start_background_tasks, process_loop

# Initialize background tasks when the application starts
print("[INFO] Initializing CHILLER for Gunicorn deployment...")

# Start background tasks
start_background_tasks()

# Start real-time processing thread
processing_thread = threading.Thread(target=process_loop, daemon=True)
processing_thread.start()
print("[INFO] Real-time processing thread started")

# Export the SocketIO app for Gunicorn
application = socketio

if __name__ == "__main__":
    # This won't be called when using Gunicorn, but useful for testing
    socketio.run(app, host="0.0.0.0", port=8000, debug=False)
