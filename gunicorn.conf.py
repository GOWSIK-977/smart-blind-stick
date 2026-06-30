# gunicorn.conf.py
import os
import multiprocessing

# Bind to port from environment
port = os.environ.get('PORT', '10000')
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1
worker_class = 'sync'
timeout = 120

# Memory optimization
max_requests = 50
max_requests_jitter = 10
preload_app = True

# Limit request size
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Use /dev/shm for temp files (if available)
worker_tmp_dir = '/dev/shm'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
