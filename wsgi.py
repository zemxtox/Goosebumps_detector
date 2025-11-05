#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn deployment with Flask-SocketIO
"""

# Import eventlet and monkey patch BEFORE importing anything else
import eventlet
eventlet.monkey_patch()

import os
import threading
import time

def create_application():
    """Create and initialize the WSGI application."""
    # Import after monkey patching
    from chiller import app, socketio, start_background_tasks, process_loop
    
    print("[INFO] Initializing CHILLER for Gunicorn deployment...")
    
    # Initialize background tasks
    try:
        start_background_tasks()
        print("[INFO] Background tasks started")
    except Exception as e:
        print(f"[ERROR] Failed to start background tasks: {e}")
    
    # Start real-time processing thread
    try:
        processing_thread = threading.Thread(target=process_loop, daemon=True)
        processing_thread.start()
        print("[INFO] Real-time processing thread started")
    except Exception as e:
        print(f"[ERROR] Failed to start processing thread: {e}")
    
    # Give a moment for initialization
    time.sleep(0.1)
    
    print("[INFO] CHILLER initialization complete")
    
    # Return the SocketIO WSGI application
    return socketio

# Create the application instance
application = create_application()

if __name__ == "__main__":
    # For testing the WSGI app directly
    from chiller import app
    application.run(app, host="0.0.0.0", port=8000, debug=False)
