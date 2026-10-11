"""Small localhost-only HTTP API using the Python standard library."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .audit import AuditLog
from .controller import ADAPTERS, ConversationController, SessionState


MAX_REQUEST_BYTES = 64 * 1024


class SessionAPI:
    """Thread-safe in-memory session store used by the local HTTP server."""

    def __init__(self, controller: ConversationController | None = None, runtime_status_provider=None):
        self.controller = controller or ConversationController(audit=AuditLog())
        self.runtime_status_provider = runtime_status_provider or self._fallback_runtime_status
        self.sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _fallback_runtime_status(adapter: str) -> dict[str, Any]:
        return {"adapter": adapter, "lifecycle_status": "unavailable",
                "mode": "deterministic_fallback", "available": False,
                "provenance_status": "unavailable"}

    def runtime_status(self, adapter: str) -> dict[str, Any]:
        status = self.runtime_status_provider(adapter)
        if not isinstance(status, Mapping):
            raise ValueError("Runtime status is unavailable")
        lifecycle = status.get("lifecycle_status")
        mode = status.get("mode")
        provenance = status.get("provenance_status")
        if (lifecycle not in {"loading", "ready", "available", "configured", "unavailable", "incompatible"}
                or mode not in {"local_model", "deterministic_fallback"}
                or provenance not in {"complete", "partial", "pending", "unavailable", "error"}):
            raise ValueError("Runtime status is unavailable")
        return {"adapter": adapter, "lifecycle_status": lifecycle, "mode": mode,
                "available": bool(status.get("available")), "provenance_status": provenance}

    def create(self, adapter: str = "sft") -> tuple[str, SessionState]:
        state = self.controller.new_session(adapter)
        session_id = uuid4().hex
        with self._lock:
            self.sessions[session_id] = state
        return session_id, state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self.sessions.get(session_id)
        if state is None:
            raise KeyError(session_id)
        return state

    def reset(self, session_id: str) -> SessionState:
        with self._lock:
            state = self.sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            replacement = self.controller.new_session(state.adapter)
            self.sessions[session_id] = replacement
        return replacement

    def message(self, session_id: str, message: str) -> dict[str, Any]:
        # The controller mutates a session, so keep lookup and handling atomic.
        with self._lock:
            state = self.sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            return self.controller.handle(state, message)


def _session_payload(session_id: str, state: SessionState) -> dict[str, Any]:
    payload = {
        item.name: _json_safe(getattr(state, item.name))
        for item in fields(state)
    }
    payload.pop("last_search", None)
    payload["session_id"] = session_id
    payload["simulated"] = True
    return payload


def _json_safe(value: Any) -> Any:
    """Convert state snapshots to JSON values without thawing them in place."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def make_server(host: str = "127.0.0.1", port: int = 8000, api: SessionAPI | None = None):
    api = api or SessionAPI()

    class Handler(BaseHTTPRequestHandler):
        server_version = "RLAIFLocal/1.0"

        def _json(self, status: int, payload: Any, *, headers: dict[str, str] | None = None):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str):
            return self._json(status, {"error": {"code": code, "message": message}})

        def _read_json(self) -> dict[str, Any]:
            if self.headers.get_content_type() != "application/json":
                raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type",
                                   "Content-Type must be application/json")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise RequestError(HTTPStatus.LENGTH_REQUIRED, "length_required",
                                   "Content-Length is required")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_content_length",
                                   "Content-Length must be an integer") from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large",
                                   f"Request body must not exceed {MAX_REQUEST_BYTES} bytes")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_json",
                                   "Request body must contain valid UTF-8 JSON") from exc
            if not isinstance(payload, dict):
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_body",
                                   "Request body must be a JSON object")
            return payload

        def _route(self) -> tuple[list[str], dict[str, list[str]]]:
            parsed = urlparse(self.path)
            return [part for part in parsed.path.split("/") if part], parse_qs(parsed.query)

        def do_GET(self):  # noqa: N802
            parts, _ = self._route()
            if not parts:
                return self._html()
            if parts == ["health"]:
                return self._json(HTTPStatus.OK, {"status": "ok", "simulator": True})
            if parts == ["adapters"]:
                return self._json(HTTPStatus.OK, {"adapters": ADAPTERS})
            if parts == ["runtime-status"]:
                adapter = parse_qs(urlparse(self.path).query).get("adapter", ["sft"])[0]
                if adapter not in ADAPTERS:
                    return self._error(HTTPStatus.BAD_REQUEST, "invalid_adapter", "Unknown adapter")
                try:
                    return self._json(HTTPStatus.OK, api.runtime_status(adapter))
                except Exception:
                    return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "runtime_status_unavailable",
                                       "Runtime status is temporarily unavailable")
            if len(parts) == 2 and parts[0] == "sessions":
                try:
                    state = api.get(parts[1])
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, "session_not_found", "Unknown session")
                return self._json(HTTPStatus.OK, _session_payload(parts[1], state))
            return self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found")

        def _html(self):
            body = _BROWSER_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            parts, query = self._route()
            try:
                payload = self._read_json()
                if parts == ["sessions"]:
                    adapter = payload.get("adapter", "sft")
                    if not isinstance(adapter, str):
                        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_adapter",
                                           "adapter must be a string")
                    session_id, state = api.create(adapter)
                    return self._json(HTTPStatus.CREATED, _session_payload(session_id, state),
                                      headers={"Location": f"/sessions/{session_id}"})
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "messages":
                    if "message" not in payload:
                        raise RequestError(HTTPStatus.BAD_REQUEST, "missing_message",
                                           "message is required")
                    result = api.message(parts[1], payload["message"])
                    wants_stream = query.get("stream") == ["1"] or "text/event-stream" in self.headers.get("Accept", "")
                    if wants_stream:
                        return self._event_stream(result)
                    return self._json(HTTPStatus.OK, result)
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "reset":
                    state = api.reset(parts[1])
                    response = _session_payload(parts[1], state)
                    response["reset"] = True
                    return self._json(HTTPStatus.OK, response)
                return self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found")
            except RequestError as exc:
                return self._error(exc.status, exc.code, exc.message)
            except KeyError:
                return self._error(HTTPStatus.NOT_FOUND, "session_not_found", "Unknown session")
            except ValueError as exc:
                return self._error(HTTPStatus.BAD_REQUEST, "validation_error", str(exc))
            except Exception:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error",
                                   "The local server could not process the request")

        def _event_stream(self, result: dict[str, Any]):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            text = result.get("text", "")
            for chunk in _text_chunks(text):
                event = json.dumps({"delta": chunk}, ensure_ascii=False)
                self.wfile.write(f"event: delta\ndata: {event}\n\n".encode("utf-8"))
                self.wfile.flush()
            final = json.dumps(result, ensure_ascii=False)
            self.wfile.write(f"event: done\ndata: {final}\n\n".encode("utf-8"))
            self.wfile.flush()
            self.close_connection = True

        def do_PUT(self):  # noqa: N802
            self._method_not_allowed()

        def do_DELETE(self):  # noqa: N802
            self._method_not_allowed()

        def do_PATCH(self):  # noqa: N802
            self._method_not_allowed()

        def _method_not_allowed(self):
            return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed",
                               "Method not allowed")

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.api = api
    return server


class RequestError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _text_chunks(text: str, size: int = 24):
    """Yield bounded text chunks without changing the response text."""
    for start in range(0, len(text), size):
        yield text[start:start + size]


_BROWSER_HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLAIF local agent</title>
<style>body{font:16px system-ui;max-width:760px;margin:40px auto;padding:0 16px}#log{min-height:300px;border:1px solid #ddd;padding:16px;overflow:auto}input{width:78%;padding:10px}button,select{padding:10px}p{white-space:pre-wrap}.meta{color:#666;font-size:12px}.error{color:#a00}.flight-options{padding-left:1.5rem}.flight-options li{margin:.35rem 0}</style></head>
<body><h1>RLAIF flight assistant</h1><p class="meta">Local simulator only — no airline or payment transactions.</p>
<label>Adapter <select id="adapter"><option>sft</option><option>dpo</option><option>ppo_checkpoint_19</option><option>ppo_adaptive_50</option></select></label>
<p id="runtime-status" class="meta" role="status">Checking selected adapter…</p>
<button id="reset" type="button">Reset session</button><div id="log" aria-live="polite"></div>
<form id="form"><input id="msg" autocomplete="off" placeholder="Find a flight from IAD to LAS" required><button>Send</button></form>
<script>
let sid,sessionAdapter=null,switching=false;const log=document.getElementById('log'),msg=document.getElementById('msg'),adapter=document.getElementById('adapter'),runtimeStatus=document.getElementById('runtime-status');
function add(w,t,cls){const p=document.createElement('p'),b=document.createElement('b'),span=document.createElement('span');b.textContent=w+': ';span.textContent=t;p.append(b,span);if(cls)p.className=cls;log.appendChild(p);log.scrollTop=log.scrollHeight;return span}
async function jsonRequest(url,options){const r=await fetch(url,options);let x;try{x=await r.json()}catch(_){throw new Error('Server returned an invalid response')}if(!r.ok)throw new Error(x.error?.message||'Request failed');return x}
async function refreshStatus(){const selected=adapter.value;try{const s=await jsonRequest('/runtime-status?adapter='+encodeURIComponent(selected));if(selected!==adapter.value)return;const states={loading:'Loading local model',ready:'Local model ready',available:'Local model available',configured:'Configured · deterministic fallback (model files unavailable)',unavailable:'Unavailable · deterministic fallback',incompatible:'Incompatible · deterministic fallback'};const provenance={complete:'provenance complete',partial:'provenance partial',pending:'provenance loading',unavailable:'provenance unavailable',error:'provenance unavailable'};runtimeStatus.textContent='Selected adapter: '+s.adapter+' · '+(states[s.lifecycle_status]||'Runtime status unavailable')+' · '+(provenance[s.provenance_status]||'provenance status unavailable');runtimeStatus.dataset.mode=s.mode==='local_model'?'local_model':'deterministic_fallback'}catch(_){if(selected===adapter.value){runtimeStatus.textContent='Selected adapter: '+selected+' · Runtime status unavailable';runtimeStatus.dataset.mode='deterministic_fallback'}}}
function showFlights(results){if(!Array.isArray(results)||!results.length)return;const list=document.createElement('ul');list.className='flight-options';list.setAttribute('aria-label','Flight options');for(const f of results){const item=document.createElement('li');const origin=f.origin||f.departure_airport,destination=f.destination||f.return_airport;item.textContent=[f.airline||'Flight',f.flight_number?'flight '+f.flight_number:'',origin&&destination?origin+' to '+destination:'',f.price!=null?'$'+f.price:'',f.departure_time||''].filter(Boolean).join(' · ');list.appendChild(item)}log.appendChild(list)}
async function reset(){switching=true;sid=null;sessionAdapter=null;adapter.disabled=true;try{const selected=adapter.value;const x=await jsonRequest('/sessions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({adapter:selected})});if(selected!==adapter.value)return;sid=x.session_id;sessionAdapter=x.adapter;log.replaceChildren();add('system','New simulated session');await refreshStatus()}catch(_){add('error','Could not start a session. Please try again.','error')}finally{adapter.disabled=false;switching=false}}
async function send(e){e.preventDefault();const m=msg.value.trim();if(!m||switching||!sid||sessionAdapter!==adapter.value)return;msg.value='';add('you',m);const target=add('agent','');try{const r=await fetch('/sessions/'+encodeURIComponent(sid)+'/messages?stream=1',{method:'POST',headers:{'content-type':'application/json','accept':'text/event-stream'},body:JSON.stringify({message:m})});if(!r.ok){let x={};try{x=await r.json()}catch(_){}throw new Error(x.error?.message||'Request failed')}const reader=r.body.getReader(),decoder=new TextDecoder();let pending='',finalResult=null;while(true){const part=await reader.read();if(part.done)break;pending+=decoder.decode(part.value,{stream:true});const events=pending.split('\\n\\n');pending=events.pop();for(const event of events){const line=event.split('\\n').find(v=>v.startsWith('data: '));if(!line)continue;const data=JSON.parse(line.slice(6));if(event.startsWith('event: delta'))target.textContent+=data.delta;else if(event.startsWith('event: done'))finalResult=data}}if(finalResult){showFlights(finalResult.results);add('status',finalResult.model_used?'Response: local model':'Response: deterministic fallback')}}catch(_){target.textContent='Could not complete the request. Please try again.';target.parentElement.className='error'}}
adapter.addEventListener('change',()=>{refreshStatus();reset()});document.getElementById('reset').addEventListener('click',reset);document.getElementById('form').addEventListener('submit',send);reset();
</script></body></html>'''
