#!/usr/bin/env python3
"""
HTTPS version of CHILLER server for PWA development
"""
import ssl
import os
from chiller import app, socketio

# SSL Configuration
HOST = "localhost"
PORT = 8000
CERT_FILE = "localhost.crt"
KEY_FILE = "localhost.key"

def create_ssl_context():
    """Create SSL context for HTTPS"""
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        print("❌ SSL certificates not found!")
        print("Run: python generate_cert.py")
        return None
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    return context

if __name__ == "__main__":
    print("="*80)
    print(" CHILLER - Goosebump Detection System (HTTPS)")
    print(" Detection: GRAYSCALE (research-validated)")
    print(" Display: COLOR (enhanced visual feedback)")
    print("="*80)
    
    ssl_context = create_ssl_context()
    if ssl_context is None:
        print("❌ Cannot start HTTPS server without SSL certificates")
        print("Run: python generate_cert.py")
        exit(1)
    
    print(f" 🔒 HTTPS Dashboard: https://{HOST}:{PORT}")
    print(f" 📱 PWA Installation: Available")
    print(f" 📷 Camera Access: Available")
    print("="*80)
    print(" 🚨 Security Warning:")
    print(" • Browser will show security warning (self-signed certificate)")
    print(" • Click 'Advanced' → 'Proceed to localhost (unsafe)'")
    print(" • This is normal for development")
    print("="*80)
    
    try:
        # Run the server with SSL
        socketio.run(
            app, 
            host=HOST, 
            port=PORT, 
            debug=False, 
            ssl_context=ssl_context,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"❌ Failed to start HTTPS server: {e}")
        print("Try running: python chiller.py (HTTP version)")