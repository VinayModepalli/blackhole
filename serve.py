"""Dev server with caching disabled so module edits show up on reload."""
import http.server
import functools

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

if __name__ == "__main__":
    handler = functools.partial(NoCacheHandler, directory=__file__.rsplit("/", 1)[0])
    http.server.ThreadingHTTPServer(("127.0.0.1", 8741), handler).serve_forever()
