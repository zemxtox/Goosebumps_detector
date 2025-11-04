#!/usr/bin/env python3
"""
Generate self-signed SSL certificate for HTTPS development
"""
import os
import subprocess
import sys

def generate_self_signed_cert():
    """Generate self-signed certificate for localhost"""
    
    # Check if OpenSSL is available
    try:
        subprocess.run(['openssl', 'version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ OpenSSL not found. Please install OpenSSL first.")
        print("Windows: Download from https://slproweb.com/products/Win32OpenSSL.html")
        print("Or use: winget install OpenSSL.Light")
        return False
    
    cert_file = 'localhost.crt'
    key_file = 'localhost.key'
    
    # Check if certificates already exist
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"✅ Certificates already exist: {cert_file}, {key_file}")
        return True
    
    print("🔐 Generating self-signed certificate for localhost...")
    
    # Generate private key and certificate
    cmd = [
        'openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-keyout', key_file,
        '-out', cert_file, '-days', '365', '-nodes', '-subj',
        '/C=US/ST=State/L=City/O=Organization/CN=localhost'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Certificate generated: {cert_file}")
        print(f"✅ Private key generated: {key_file}")
        print("\n📝 To use HTTPS:")
        print("1. Run: python chiller_https.py")
        print("2. Visit: https://localhost:8000")
        print("3. Accept the security warning (self-signed certificate)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate certificate: {e}")
        return False

if __name__ == "__main__":
    generate_self_signed_cert()