"""Characterize the public routes registered by the FastAPI application."""

import unittest

from server import app
from ui_routes import router as ui_router


class ServerRouteTests(unittest.TestCase):
    def test_existing_route_paths_and_methods_remain_registered(self):
        routes = {
            (method.lower(), path)
            for path, operations in app.openapi()["paths"].items()
            for method in operations
        }

        self.assertTrue(
            {
                ("get", "/api/model-info"),
                ("get", "/api/ui/initial-data"),
                ("post", "/save_settings"),
                ("post", "/restart_server"),
                ("post", "/tts"),
                ("post", "/v1/audio/speech"),
            }.issubset(routes)
        )
        self.assertTrue(any(route.path == "/" for route in ui_router.routes))


if __name__ == "__main__":
    unittest.main()
