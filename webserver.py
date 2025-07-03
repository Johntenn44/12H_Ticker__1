from http.server import HTTPServer, BaseHTTPRequestHandler
import os

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)  # HTTP status 200 OK
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is running")  # Simple response body

def run():
    # Use port from environment variable PORT or default to 8000
    port = int(os.environ.get("PORT", 8000))
    server_address = ('0.0.0.0', port)  # Bind to all interfaces
    httpd = HTTPServer(server_address, SimpleHandler)
    print(f"Starting HTTP server on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
