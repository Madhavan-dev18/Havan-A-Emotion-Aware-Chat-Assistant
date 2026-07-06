import os

# Gunicorn Production Configuration
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
timeout = 120

# Preload application code and models in the master process before worker processes are forked.
# This allows child processes to share the loaded neural network weights (CoW) and prevents 
# request worker blocking on initial routes.
preload_app = True
