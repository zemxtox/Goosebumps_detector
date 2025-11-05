#!/usr/bin/env python3
"""
Direct production runner for CHILLER with proper Flask-SocketIO + Gunicorn setup
"""
import eventlet
eventlet.monkey_patch()

import os
import sys
import threading
from chiller import app, socketio, start_background_tasks, process_loop

def main():
    """Run the application in production mode."""
    print("🚀 Starting CHILLER in Production Mode")
    print("=" * 50)
    
    # Get configuration
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"⚙️ Async Mode: eventlet")
    print("=" * 50)
    
    # Initialize background tasks
    print("[INFO] Starting background tasks...")
    start_background_tasks()
    
    # Start real-time processing thread
    print("[INFO] Starting real-time processing thread...")
    processing_thread = threading.Thread(target=process_loop, daemon=True)
    processing_thread.start()
    
    print("[INFO] All systems ready!")
    print("=" * 50)
    
    # Run with SocketIO (which handles Gunicorn integration automatically)
    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
