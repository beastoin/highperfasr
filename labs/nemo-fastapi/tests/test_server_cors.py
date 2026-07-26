import importlib
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_server(monkeypatch, origins):
    if origins is None:
        monkeypatch.delenv("HPFASR_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("HPFASR_CORS_ORIGINS", origins)

    import highperfasr.server as server

    return importlib.reload(server)


def cors_middleware(server):
    return [middleware for middleware in server.app.user_middleware if middleware.cls.__name__ == "CORSMiddleware"]


def test_cors_disabled_by_default(monkeypatch):
    server = load_server(monkeypatch, None)

    assert cors_middleware(server) == []


def test_cors_origins_enabled_from_env(monkeypatch):
    server = load_server(monkeypatch, "https://example.com, http://localhost:5173 ")

    middleware = cors_middleware(server)
    assert len(middleware) == 1
    assert middleware[0].kwargs["allow_origins"] == ["https://example.com", "http://localhost:5173"]


def test_cors_wildcard_is_normalized(monkeypatch):
    server = load_server(monkeypatch, "https://example.com,*")

    middleware = cors_middleware(server)
    assert len(middleware) == 1
    assert middleware[0].kwargs["allow_origins"] == ["*"]
