from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter("ml_api_requests_total", "Total API requests", ["method", "path", "status_code"])
PREDICTIONS_TOTAL = Counter("ml_predictions_total", "Total predictions", ["predicted_class"])
ERRORS_TOTAL = Counter("ml_api_errors_total", "Total API errors", ["path"])
REQUEST_LATENCY_SECONDS = Histogram("ml_api_request_latency_seconds", "Request latency", ["method", "path"])
