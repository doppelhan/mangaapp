import json
import pytest
from app import app


def test_root_get():
    with app.test_client() as client:
        resp = client.get('/')
        assert resp.status_code == 200


def test_process_without_image():
    with app.test_client() as client:
        resp = client.post('/process', json={})
        assert resp.status_code == 200
        assert resp.get_json() == {'status': 'error'}

