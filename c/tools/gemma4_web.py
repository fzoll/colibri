#!/usr/bin/env python3
"""Minimal web chat UI for Gemma 4 via Colibri engine.
Opens on port 8080 — access from phone at http://192.168.68.61:8080

Usage:
    python3 tools/gemma4_web.py --snap ./gemma4_26b_i8 --vulkan
"""
import sys, os, json, subprocess, tempfile, argparse, html
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
.msg { margin: 8px 0; padding: 10px 14px; border-radius: 12px; max-width: 85%; line-height: 1.5; }
.user { background: #16213e; margin-left: auto; text-align: right; }
.bot { background: #0f3460; white-space: pre-wrap; }
.bot b, .bot strong { color: #e94560; }
.bot h3, .bot h2 { color: #e94560; margin: 8px 0 4px; font-size: 15px; }
#form { display: flex; gap: 8px; padding: 10px 0; }
#inp { flex: 1; padding: 12px; border-radius: 20px; border: 1px solid #333; background: #16213e; color: #eee; font-size: 16px; }
#btn { padding: 12px 20px; border-radius: 20px; border: none; background: #e94560; color: white; font-weight: bold; font-size: 16px; cursor: pointer; }
#btn:disabled { opacity: 0.5; }
.spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #333; border-top: 3px solid #e94560; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { 100% { transform: rotate(360deg); } }
</style></head><body>
<h1>Gemma 4 26B on RPi 5</h1>
<div class="sub">Colibri engine + Vulkan GPU</div>
<div id="chat"></div>
<form id="form" onsubmit="send(); return false;">
<input id="inp" placeholder="Type a message..." autocomplete="off">
<button id="btn" type="submit">Send</button>
</form>
<script>
function send() {
  var inp = document.getElementById('inp');
  var msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  var chat = document.getElementById('chat');
  chat.innerHTML += '<div class="msg user">' + esc(msg) + '</div>';
  chat.innerHTML += '<div class="msg bot" id="loading"><div class="spinner"></div> Thinking...</div>';
  chat.scrollTop = chat.scrollHeight;
  document.getElementById('btn').disabled = true;
  fetch('/chat', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'q='+encodeURIComponent(msg)})
  .then(r => r.text())
  .then(t => {
    document.getElementById('loading').outerHTML = '<div class="msg bot">' + fmt(t) + '</div>';
    document.getElementById('btn').disabled = false;
    chat.scrollTop = chat.scrollHeight;
  })
  .catch(e => {
    document.getElementById('loading').outerHTML = '<div class="msg bot">Error: ' + esc(e.toString()) + '</div>';
    document.getElementById('btn').disabled = false;
  });
}
function esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function fmt(s) {
  s = esc(s);
  s = s.replace(/\\*\\*(.+?)\\*\\*/g, '<b>$1</b>');
  s = s.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  s = s.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  s = s.replace(/\\n/g, '<br>');
  return s;
}
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode()
        params = parse_qs(body)
        query = params.get('q', [''])[0]

        reply = generate(query)

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(reply.encode())

    def log_message(self, format, *args):
        pass  # suppress request logs

def generate(user_msg):
    prompt = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
    ids = TOK.encode(prompt).ids

    full_ids = list(ids) + [0] * ARGS.ngen
    tf_pred = [0] * len(full_ids)
    ref = {"prompt_ids": list(ids), "full_ids": full_ids, "tf_pred": tf_pred}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ref, f)
        ref_path = f.name

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
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return "[timeout — try a shorter question]"
    finally:
        os.unlink(ref_path)

    output_ids = []
    for line in result.stdout.split('\n'):
        if line.startswith("Motore C Gemma4"):
            parts = line.split(':')
            if len(parts) >= 2:
                nums = parts[-1].strip().split()
                output_ids = []
                for x in nums:
                    try: output_ids.append(int(x))
                    except ValueError: pass

    if output_ids:
        filtered = [x for x in output_ids if x > 0 and x != 1]
        text = TOK.decode(filtered)
        # Remove "thought\n" prefix if present
        if text.startswith("thought\n"):
            text = text[len("thought\n"):]
        elif text.startswith("thought"):
            text = text[len("thought"):]
        return text.strip()
    return "[no output — check engine logs]"

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
    parser.add_argument("--port", type=int, default=8080)
    ARGS = parser.parse_args()

    from tokenizers import Tokenizer
    TOK = Tokenizer.from_file(os.path.join(ARGS.snap, "tokenizer.json"))

    print(f"Gemma 4 Web Chat — http://0.0.0.0:{ARGS.port}")
    print(f"Open on phone: http://192.168.68.61:{ARGS.port}")
    HTTPServer(('0.0.0.0', ARGS.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
