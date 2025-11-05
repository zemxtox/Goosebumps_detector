#!/usr/bin/env python3
"""
Gunicorn configuration for CHILLER production deployment
"""
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
backlog = 2048

# Worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', 1))  # Single worker for Socket.IO
worker_class = "eventlet"  # Required for Socket.IO
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "chiller-goosebump-detector"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if certificates are available)
keyfile = None
certfile = None

# Performance tuning
preload_app = True  # Load application code before forking workers
enable_stdio_inheritance = True

# Environment variables
raw_env = [
    f"DETECTION_THRESHOLD={os.environ.get('DETECTION_THRESHOLD', '25.0')}",
    f"VIDEO_DETECTION_THRESHOLD={os.environ.get('VIDEO_DETECTION_THRESHOLD', '60.0')}",
    f"BASELINE_FRAMES={os.environ.get('BASELINE_FRAMES', '5')}",
    f"SAVE_DETECTIONS={os.environ.get('SAVE_DETECTIONS', 'True')}",
]

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("CHILLER server is ready. Accepting connections.")

def worker_int(worker):
    """Called just after a worker has been killed by SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")
