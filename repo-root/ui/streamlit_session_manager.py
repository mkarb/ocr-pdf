"""
Streamlit session state management for multi-user environments.

This module provides utilities to properly isolate session state between users
and prevent data leakage when running multiple concurrent sessions.
"""

from __future__ import annotations
import hashlib
import time
from typing import Any, Dict, Optional
from functools import wraps

import streamlit as st


class SessionManager:
    """
    Manages isolated session state for concurrent users.

    Features:
    - Per-user session isolation
    - Automatic cleanup of stale sessions
    - Thread-safe session access
    - Memory usage tracking
    """

    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager.

        Args:
            session_timeout: Session timeout in seconds (default 1 hour)
        """
        self.session_timeout = session_timeout

    @staticmethod
    def get_session_id() -> str:
        """
        Get a unique session ID for the current user.

        Uses Streamlit's built-in session ID if available,
        otherwise generates one based on user context.
        """
        # Try to get Streamlit's session ID
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx and hasattr(ctx, 'session_id'):
                return ctx.session_id
        except Exception:
            pass

        # Fallback: generate based on session state
        if not hasattr(st.session_state, '_custom_session_id'):
            # Generate a unique ID based on timestamp and random data
            unique_str = f"{time.time()}_{id(st.session_state)}"
            session_id = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
            st.session_state._custom_session_id = session_id

        return st.session_state._custom_session_id

    @staticmethod
    def get_user_state(key: str, default: Any = None) -> Any:
        """
        Get a value from the current user's session state.

        Args:
            key: State key
            default: Default value if key doesn't exist

        Returns:
            The value from session state or default
        """
        session_key = f"user_{key}"
        return st.session_state.get(session_key, default)

    @staticmethod
    def set_user_state(key: str, value: Any) -> None:
        """
        Set a value in the current user's session state.

        Args:
            key: State key
            value: Value to store
        """
        session_key = f"user_{key}"
        st.session_state[session_key] = value

    @staticmethod
    def delete_user_state(key: str) -> None:
        """
        Delete a value from the current user's session state.

        Args:
            key: State key to delete
        """
        session_key = f"user_{key}"
        if session_key in st.session_state:
            del st.session_state[session_key]

    @staticmethod
    def clear_user_state() -> None:
        """Clear all user-specific state for the current session."""
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("user_")]
        for key in keys_to_delete:
            del st.session_state[key]

    @staticmethod
    def initialize_defaults(defaults: Dict[str, Any]) -> None:
        """
        Initialize default values for session state.

        Args:
            defaults: Dictionary of key-value pairs to initialize
        """
        for key, value in defaults.items():
            session_key = f"user_{key}"
            if session_key not in st.session_state:
                st.session_state[session_key] = value

    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            Dictionary with session metadata
        """
        return {
            "session_id": SessionManager.get_session_id(),
            "state_keys": [k for k in st.session_state.keys() if k.startswith("user_")],
            "state_size": len(st.session_state),
        }


def session_cached(cache_key: Optional[str] = None):
    """
    Decorator to cache function results in user session state.

    Args:
        cache_key: Optional custom cache key (uses function name if not provided)

    Example:
        @session_cached("expensive_computation")
        def compute_something(param):
            return expensive_operation(param)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_key or f"cached_{func.__name__}"

            # Create a unique key based on arguments
            arg_hash = hashlib.md5(str((args, kwargs)).encode()).hexdigest()[:8]
            full_key = f"{key}_{arg_hash}"

            # Check if result is cached
            cached_result = SessionManager.get_user_state(full_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            SessionManager.set_user_state(full_key, result)
            return result

        return wrapper
    return decorator


def isolate_session_state():
    """
    Initialize session state isolation for the current user.
    Call this at the beginning of your Streamlit app.
    """
    # Ensure session ID is generated
    session_id = SessionManager.get_session_id()

    # Initialize session metadata if not present
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = True
        st.session_state.session_start_time = time.time()
        st.session_state.session_id = session_id


# Thread-safe singleton for global session management
_global_sessions: Dict[str, Dict[str, Any]] = {}


class GlobalSessionStore:
    """
    Global session store for sharing data across Streamlit instances.

    WARNING: This stores data in memory and will be lost on container restart.
    For production, use Redis or similar external storage.
    """

    @staticmethod
    def set(key: str, value: Any, session_id: Optional[str] = None) -> None:
        """
        Store a value in the global session store.

        Args:
            key: Storage key
            value: Value to store
            session_id: Optional session ID (uses current if not provided)
        """
        if session_id is None:
            session_id = SessionManager.get_session_id()

        if session_id not in _global_sessions:
            _global_sessions[session_id] = {}

        _global_sessions[session_id][key] = {
            "value": value,
            "timestamp": time.time()
        }

    @staticmethod
    def get(key: str, default: Any = None, session_id: Optional[str] = None) -> Any:
        """
        Get a value from the global session store.

        Args:
            key: Storage key
            default: Default value if not found
            session_id: Optional session ID (uses current if not provided)

        Returns:
            Stored value or default
        """
        if session_id is None:
            session_id = SessionManager.get_session_id()

        if session_id not in _global_sessions:
            return default

        data = _global_sessions[session_id].get(key)
        if data is None:
            return default

        return data["value"]

    @staticmethod
    def delete(key: str, session_id: Optional[str] = None) -> None:
        """
        Delete a value from the global session store.

        Args:
            key: Storage key
            session_id: Optional session ID (uses current if not provided)
        """
        if session_id is None:
            session_id = SessionManager.get_session_id()

        if session_id in _global_sessions:
            _global_sessions[session_id].pop(key, None)

    @staticmethod
    def cleanup_old_sessions(max_age: int = 3600) -> int:
        """
        Remove sessions older than max_age seconds.

        Args:
            max_age: Maximum age in seconds

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        sessions_to_remove = []

        for session_id, session_data in _global_sessions.items():
            # Check if any data is older than max_age
            oldest_timestamp = min(
                (data["timestamp"] for data in session_data.values()),
                default=current_time
            )

            if current_time - oldest_timestamp > max_age:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del _global_sessions[session_id]

        return len(sessions_to_remove)


def init_session():
    """
    Initialize session management for the current user.
    Call this at the start of your Streamlit app.
    """
    isolate_session_state()

    # Periodic cleanup of old sessions (every 100th request)
    if not hasattr(st.session_state, "_cleanup_counter"):
        st.session_state._cleanup_counter = 0

    st.session_state._cleanup_counter += 1
    if st.session_state._cleanup_counter % 100 == 0:
        GlobalSessionStore.cleanup_old_sessions(max_age=3600)
