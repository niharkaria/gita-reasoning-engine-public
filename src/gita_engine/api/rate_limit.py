"""Shared rate limiter instance.

Why this file exists:
    Both main.py (to attach the limiter to app.state) and routes.py (to
    apply @limiter.limit(...) to specific endpoints) need the same
    Limiter instance. Defining it in either of those two files and
    importing it into the other creates a circular import, since
    main.py also imports routes.router. This standalone module breaks
    that cycle.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter: Limiter = Limiter(key_func=get_remote_address)
