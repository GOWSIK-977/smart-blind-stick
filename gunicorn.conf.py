# gunicorn.conf.py
import os

# Force bind to port 10000 (Render's default)
bind = "0.0.0.0:10000"

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

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
