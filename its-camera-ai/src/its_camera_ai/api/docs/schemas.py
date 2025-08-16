"""
OpenAPI schema generation and API documentation examples.

Provides comprehensive API documentation with examples and schemas.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


class APIExamples:
    """API documentation examples for all endpoints."""

    @staticmethod
    def get_auth_examples() -> dict[str, Any]:
        """Get authentication endpoint examples."""
        return {
            "login": {
                "summary": "Successful login",
                "description": "Login with valid credentials",
                "value": {
                    "username": "admin@example.com",
                    "password": "SecurePassword123!",
                    "remember_me": True,
                },
            },
            "mfa_login": {
                "summary": "Login with MFA",
                "description": "Login requiring multi-factor authentication",
                "value": {
                    "username": "admin@example.com",
                    "password": "SecurePassword123!",
                    "mfa_code": "123456",
                },
            },
        }

    @staticmethod
    def get_camera_examples() -> dict[str, Any]:
        """Get camera endpoint examples."""
        return {
            "create_camera": {
                "summary": "Add IP camera",
                "description": "Register a new IP camera stream",
                "value": {
                    "name": "Highway Exit 42 Camera",
                    "location": {"lat": 37.7749, "lng": -122.4194},
                    "stream_url": "rtsp://192.168.1.100:554/stream",
                    "resolution": "1920x1080",
                    "fps": 30,
                    "tags": ["highway", "exit", "high-traffic"],
                },
            },
            "update_settings": {
                "summary": "Update camera settings",
                "value": {
                    "resolution": "3840x2160",
                    "fps": 60,
                    "enable_motion_detection": True,
                    "motion_sensitivity": 0.8,
                },
            },
        }

    @staticmethod
    def get_model_examples() -> dict[str, Any]:
        """Get ML model endpoint examples."""
        return {
            "register_model": {
                "summary": "Register YOLO11 model",
                "description": "Register a new traffic detection model",
                "value": {
                    "name": "YOLO11-Traffic-v2",
                    "version": "2.0.0",
                    "framework": "pytorch",
                    "task": "object_detection",
                    "metrics": {
                        "mAP": 0.92,
                        "precision": 0.94,
                        "recall": 0.91,
                        "f1_score": 0.925,
                    },
                    "metadata": {
                        "trained_on": "2024-01-15",
                        "dataset": "traffic-dataset-v3",
                        "epochs": 100,
                        "batch_size": 32,
                    },
                },
            },
            "deploy_model": {
                "summary": "Deploy to production",
                "description": "Deploy model to production stage",
                "value": {"deployment_config": {"replicas": 3, "gpu_enabled": True}},
            },
        }

    @staticmethod
    def get_analytics_examples() -> dict[str, Any]:
        """Get analytics endpoint examples."""
        return {
            "traffic_flow": {
                "summary": "Traffic flow analysis",
                "value": {
                    "time_range": "24h",
                    "metrics": ["vehicle_count", "average_speed", "congestion_level"],
                    "group_by": "hour",
                },
            },
            "predictions": {
                "summary": "Traffic predictions",
                "value": {
                    "hours_ahead": 24,
                    "confidence_threshold": 0.8,
                    "include_weather": True,
                },
            },
            "alert_rule": {
                "summary": "Create congestion alert",
                "value": {
                    "name": "High Congestion Alert",
                    "condition": "congestion_level > 0.8",
                    "actions": ["email", "sms", "dashboard_notification"],
                    "cooldown_minutes": 30,
                },
            },
        }


class OpenAPIGenerator:
    """Generate enhanced OpenAPI documentation."""

    @staticmethod
    def customize_openapi(app: FastAPI) -> dict[str, Any]:
        """Customize OpenAPI schema with examples and additional info."""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="ITS Camera AI API",
            version="3.0.0",
            description="""
# ITS Camera AI - Traffic Intelligence Platform

## Overview
Advanced AI-powered traffic monitoring and analytics system with real-time 
vehicle detection, traffic flow analysis, and predictive insights.

## Features
- 🎥 **Real-time Camera Management** - Multi-protocol support (RTSP/HTTP/WebRTC)
- 🤖 **ML Model Management** - YOLO11 deployment with <100ms inference
- 🔍 **License Plate Recognition** - Sub-15ms LPR with multi-regional support
- 📊 **Advanced Analytics** - Traffic patterns, predictions, and heatmaps
- 🚨 **Watchlist Monitoring** - Real-time alerts for plates of interest
- 🔒 **Enterprise Security** - JWT auth, MFA, RBAC, API keys
- 📈 **Scalable Architecture** - Handles 1000+ cameras, 100K+ req/s

## Authentication
The API uses JWT bearer tokens for authentication. Include the token in the 
Authorization header:
```
Authorization: Bearer <your-token>
```

## Rate Limiting
- Standard tier: 1000 requests/minute
- Premium tier: 10,000 requests/minute
- Enterprise tier: Unlimited

## Response Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## Support
- Documentation: https://docs.its-camera-ai.com
- API Status: https://status.its-camera-ai.com
- Support: support@its-camera-ai.com
            """,
            routes=app.routes,
            tags=[
                {
                    "name": "authentication",
                    "description": "User authentication and authorization",
                },
                {"name": "cameras", "description": "Camera management and control"},
                {
                    "name": "models",
                    "description": "ML model lifecycle management",
                },
                {
                    "name": "analytics",
                    "description": "Traffic analytics and insights",
                },
                {"name": "system", "description": "System monitoring and control"},
                {"name": "health", "description": "Health checks and status"},
            ],
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT authentication token",
            },
            "apiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service-to-service auth",
            },
        }

        # Add servers
        openapi_schema["servers"] = [
            {"url": "https://api.its-camera-ai.com", "description": "Production"},
            {
                "url": "https://staging-api.its-camera-ai.com",
                "description": "Staging",
            },
            {"url": "http://localhost:8000", "description": "Development"},
        ]

        # Add external docs
        openapi_schema["externalDocs"] = {
            "description": "Full API Documentation",
            "url": "https://docs.its-camera-ai.com/api",
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    @staticmethod
    def add_endpoint_examples(app: FastAPI) -> None:
        """Add request/response examples to endpoints."""
        examples = APIExamples()

        # This would be called during app initialization to add examples
        # to specific endpoints using FastAPI's built-in example system
        pass


# Singleton instance
api_examples = APIExamples()
