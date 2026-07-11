#!/usr/bin/env python3
"""Streaming web chat UI for Gemma 4 via Colibri engine.
Uses Server-Sent Events (SSE) for token-by-token streaming.

Usage:
    python3 tools/gemma4_web.py --snap ./gemma4_26b_i8 --vulkan --port 8888
"""
import sys, os, json, subprocess, tempfile, argparse, threading, time, re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

ARGS = None
TOK = None

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma 4 - RPi 5</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #eee; max-width: 600px; margin: 0 auto; padding: 10px; height: 100vh; display: flex; flex-direction: column; }
h1 { text-align: center; padding: 10px; font-size: 18px; color: #e94560; }
.sub { text-align: center; font-size: 12px; color: #888; margin-bottom: 10px; }
#chat { flex: 1; overflow-y: auto; padding: 10px; }
.msg { margin: 8px 0; padding: 10px 14px; border-radius: 12px; max-width: 85%; line-height: 1.5; word-wrap: break-word; }
.user { background: #16213e; margin-left: auto; text-align: right; }
.bot { background: #0f3460; white-space: pre-wrap; }
#form { display: flex; gap: 8px; padding: 10px 0; }
#inp { flex: 1; padding: 12px; border-radius: 20px; border: 1px solid #333; background: #16213e; color: #eee; font-size: 16px; }
#btn { padding: 12px 20px; border-radius: 20px; border: none; background: #e94560; color: white; font-weight: bold; font-size: 16px; cursor: pointer; }
#btn:disabled { opacity: 0.5; }
.typing { color: #e94560; font-style: italic; }
</style></head><body>
<h1>Gemma 4 26B on RPi 5</h1>
<div class="sub">Colibri engine + Vulkan GPU | streaming</div>
<div id="chat"></div>
<form id="form" onsubmit="send(); return false;">
<input id="inp" placeholder="Type a message..." autocomplete="off">
<button id="btn" type="submit">Send</button>
</form>
<script>
var sse = null;
function send() {
  var inp = document.getElementById('inp');
  var msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  var chat = document.getElementById('chat');
  chat.innerHTML += '<div class="msg user">' + esc(msg) + '</div>';
  var botDiv = document.createElement('div');
  botDiv.className = 'msg bot';
  botDiv.id = 'current';
  botDiv.innerHTML = '<span class="typing">thinking...</span>';
  chat.appendChild(botDiv);
  chat.scrollTop = chat.scrollHeight;
  document.getElementById('btn').disabled = true;

  if (sse) sse.close();
  sse = new EventSource('/stream?q=' + encodeURIComponent(msg));
  var text = '';
  sse.onmessage = function(e) {
    if (e.data === '[DONE]') {
      sse.close(); sse = null;
      document.getElementById('btn').disabled = false;
      return;
    }
    text += e.data;
    botDiv.textContent = text;
    chat.scrollTop = chat.scrollHeight;
  };
  sse.onerror = function() {
    sse.close(); sse = null;
    if (!text) botDiv.innerHTML = '<span class="typing">error or timeout</span>';
    document.getElementById('btn').disabled = false;
  };
}
function esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/stream'):
            self.handle_stream()
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

    def handle_stream(self):
        params = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        query = params.get('q', [''])[0]
        if not query:
            self.send_error(400)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        ids = TOK.encode(prompt).ids

        full_ids = list(ids) + [0] * ARGS.ngen
        tf_pred = [0] * len(full_ids)
        ref = {"prompt_ids": list(ids), "full_ids": full_ids, "tf_pred": tf_pred}

        ref_path = tempfile.mktemp(suffix='.json')
        with open(ref_path, 'w') as f:
            json.dump(ref, f)

        env = dict(os.environ)
        env["SNAP"] = ARGS.snap
        env["REF"] = ref_path
        if ARGS.vulkan:
            env["COLI_VULKAN"] = "1"
        if ARGS.metal:
            env["COLI_METAL"] = "1"
        if ARGS.pin:
            env["PIN"] = ARGS.pin
            env["PIN_GB"] = str(ARGS.pin_gb)

        cmd = [ARGS.engine, "8", str(ARGS.bits), str(ARGS.bits)]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Stream stderr progress lines as SSE keepalive
        def read_stderr():
            for line in proc.stderr:
                pass  # consume stderr silently

        t = threading.Thread(target=read_stderr, daemon=True)
        t.start()

        # Wait for process to finish, then decode and stream tokens
        stdout, _ = proc.communicate(timeout=600)
        output_text = stdout.decode('utf-8', errors='replace')

        try:
            os.unlink(ref_path)
        except OSError:
            pass

        # Parse output token IDs
        output_ids = []
        for line in output_text.split('\n'):
            if line.startswith("Motore C Gemma4"):
                parts = line.split(':')
                if len(parts) >= 2:
                    for x in parts[-1].strip().split():
                        try:
                            output_ids.append(int(x))
                        except ValueError:
                            pass

        if output_ids:
            filtered = [x for x in output_ids if x > 0 and x != 1]
            # Decode token by token for streaming effect
            decoded_so_far = ""
            for i in range(1, len(filtered) + 1):
                new_text = TOK.decode(filtered[:i])
                # Remove "thought" prefix
                if new_text.startswith("thought\n"):
                    new_text = new_text[len("thought\n"):]
                elif new_text.startswith("thought"):
                    new_text = new_text[len("thought"):]
                delta = new_text[len(decoded_so_far):]
                decoded_so_far = new_text
                if delta:
                    try:
                        self.wfile.write(f"data: {delta}\n\n".encode())
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
                    time.sleep(0.02)  # slight delay for visual streaming
        else:
            self.wfile.write(b"data: [no output]\n\n")
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self):
        self.send_error(405)

    def log_message(self, format, *args):
        pass

def main():
    global ARGS, TOK
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--ngen", type=int, default=200)
    parser.add_argument("--vulkan", action="store_true")
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--pin", default=None)
    parser.add_argument("--pin-gb", type=float, default=4)
    parser.add_argument("--engine", default="./gemma4")
    parser.add_argument("--port", type=int, default=8888)
    ARGS = parser.parse_args()

    from tokenizers import Tokenizer
    TOK = Tokenizer.from_file(os.path.join(ARGS.snap, "tokenizer.json"))

    print(f"Gemma 4 Streaming Chat — http://0.0.0.0:{ARGS.port}")
    HTTPServer(('0.0.0.0', ARGS.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
