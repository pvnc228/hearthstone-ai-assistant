"""
Lightweight HTTP Server for Hearthstone AI Assistant Open Source Dashboard.
Uses Python standard library (http.server) to serve web/index.html on http://127.0.0.1:8080.
"""

import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8765

def run_server(port: int = DEFAULT_PORT, auto_open: bool = True):
    web_dir = Path(__file__).parent.resolve()
    os.chdir(web_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Disable caching for fresh development view
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()

        def log_message(self, format, *args):
            # Clean minimal logging
            sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")
            sys.stdout.flush()

    for p in range(port, port + 10):
        try:
            with socketserver.TCPServer(("127.0.0.1", p), CustomHandler) as httpd:
                url = f"http://127.0.0.1:{p}/"
                print("=" * 70)
                print("  Hearthstone AI Assistant — Open Source Training Dashboard")
                print("=" * 70)
                print(f"  Server running at: {url}")
                print(f"  Serving files from: {web_dir}")
                print("  Press Ctrl+C to stop the server.")
                print("=" * 70)
                if auto_open:
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                httpd.serve_forever()
                break
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 10048:
                continue
            raise

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else DEFAULT_PORT
    no_browser = "--no-browser" in sys.argv
    try:
        run_server(port=port, auto_open=not no_browser)
    except KeyboardInterrupt:
        print("\nServer stopped.")
