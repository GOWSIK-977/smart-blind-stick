# gunicorn.conf.py
import multiprocessing

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

# Use /dev/shm for temp files
worker_tmp_dir = '/dev/shm'
