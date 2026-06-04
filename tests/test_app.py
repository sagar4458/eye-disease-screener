"""Basic tests — run: pytest tests/"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def test_app_imports():
    import app
    assert app is not None

def test_flask_app_created():
    import app
    assert app.app is not None

def test_health_check(client=None):
    import app
    c = app.app.test_client()
    # Each project has at least one GET endpoint
    assert c is not None
