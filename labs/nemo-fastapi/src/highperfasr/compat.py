"""NeMo compatibility patches — version-gated, idempotent, logged at startup.

Each patch fixes a specific NeMo issue that affects production serving.
See https://github.com/beastoin/NeMo for the patched NeMo fork.
"""

import logging

log = logging.getLogger(__name__)

_applied: set[str] = set()


def _once(name: str) -> bool:
    if name in _applied:
        return False
    _applied.add(name)
    return True


def _patch_gc_disable():
    """Disable automatic cyclic GC to prevent CUDA pinned-memory crashes.

    Under sustained load, automatic GC can trigger on the async event-loop
    thread and free CUDA pinned-memory tensors there, crashing in
    CachingHostAllocatorImpl::free(). The GPU worker calls gc.collect()
    after every batch on its own thread instead.
    """
    if not _once("gc_disable"):
        return
    import gc

    gc.disable()
    log.info("compat: disabled automatic GC (pinned-memory safety)")


def _patch_ws_upgrade():
    """Use freshly allocated bytes in WebSocket upgrade requests.

    GPU inference may corrupt Python byte constants used in HTTP request
    construction. This replaces the affected method with one that builds
    the request line from individually allocated bytes.
    """
    if not _once("ws_upgrade"):
        return
    try:
        import uvicorn.protocols.http.httptools_impl as _hi
    except ImportError:
        log.debug("compat: uvicorn httptools not available, skipping ws_upgrade patch")
        return

    def _safe_handle_websocket_upgrade(self):
        method = self.scope["method"].encode()
        url = self.url
        request_line = bytearray()
        request_line.extend(method)
        request_line.append(0x20)
        request_line.extend(url)
        request_line.append(0x20)
        request_line.extend(b"HTTP/1.1\r\n")
        for name, value in self.scope["headers"]:
            request_line.extend(name)
            request_line.append(0x3A)
            request_line.append(0x20)
            request_line.extend(value)
            request_line.append(0x0D)
            request_line.append(0x0A)
        request_line.append(0x0D)
        request_line.append(0x0A)
        protocol = self.ws_protocol_class(
            config=self.config,
            server_state=self.server_state,
            app_state=self.app_state,
        )
        protocol.connection_made(self.transport)
        protocol.data_received(bytes(request_line))
        self.transport.set_protocol(protocol)
        self.connections.discard(self)

    _hi.HttpToolsProtocol.handle_websocket_upgrade = _safe_handle_websocket_upgrade
    log.info("compat: patched uvicorn WebSocket upgrade (byte-constant safety)")


def apply_compat():
    """Apply all NeMo/runtime compatibility patches. Safe to call multiple times."""
    _patch_gc_disable()
    _patch_ws_upgrade()
