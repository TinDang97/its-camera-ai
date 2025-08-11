#!/usr/bin/env python3
"""
Database seeding script for ITS Camera AI
Populates the database with sample data for development and testing
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from pydantic import BaseSettings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class Settings(BaseSettings):
    """Database settings for seeding."""

    database_url: str = (
        "postgresql://its_user:its_password@localhost:5432/its_camera_ai"
    )

    class Config:
        env_file = ".env"


async def create_sample_cameras(conn: asyncpg.Connection) -> list[str]:
    """Create sample camera entries."""
    cameras_data = [
        {
            "name": "Downtown Intersection A",
            "location": {
                "lat": 37.7749,
                "lng": -122.4194,
                "address": "Market St & 5th St, San Francisco, CA",
            },
            "stream_url": "rtsp://demo:demo@192.168.1.100:554/stream",
            "security_level": "public",
        },
        {
            "name": "Highway Overpass B",
            "location": {
                "lat": 37.7849,
                "lng": -122.4094,
                "address": "I-80 Overpass, San Francisco, CA",
            },
            "stream_url": "rtsp://demo:demo@192.168.1.101:554/stream",
            "security_level": "internal",
        },
        {
            "name": "Shopping Center C",
            "location": {
                "lat": 37.7649,
                "lng": -122.4294,
                "address": "Union Square, San Francisco, CA",
            },
            "stream_url": "rtsp://demo:demo@192.168.1.102:554/stream",
            "security_level": "internal",
        },
        {
            "name": "Residential Area D",
            "location": {
                "lat": 37.7549,
                "lng": -122.4394,
                "address": "Castro St, San Francisco, CA",
            },
            "stream_url": "rtsp://demo:demo@192.168.1.103:554/stream",
            "security_level": "confidential",
        },
        {
            "name": "Industrial Zone E",
            "location": {
                "lat": 37.7949,
                "lng": -122.3994,
                "address": "Mission Bay, San Francisco, CA",
            },
            "stream_url": "rtsp://demo:demo@192.168.1.104:554/stream",
            "security_level": "internal",
        },
    ]

    camera_ids = []
    for camera in cameras_data:
        camera_id = await conn.fetchval(
            """
            INSERT INTO cameras (name, location, stream_url, security_level)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            camera["name"],
            camera["location"],
            camera["stream_url"],
            camera["security_level"],
        )
        if camera_id:
            camera_ids.append(camera_id)
            print(f"Created camera: {camera['name']} (ID: {camera_id})")

    return camera_ids


async def create_sample_models(conn: asyncpg.Connection) -> list[str]:
    """Create sample model registry entries."""
    models_data = [
        {
            "name": "YOLOv11n",
            "version": "1.0.0",
            "stage": "production",
            "model_path": "/models/yolo11n.pt",
            "model_format": "pytorch",
            "metrics": {
                "accuracy": 0.85,
                "latency_ms": 45,
                "map50": 0.78,
                "model_size_mb": 6.2,
            },
            "metadata": {
                "description": "YOLOv11 nano model for real-time inference",
                "input_size": [640, 640],
                "classes": 80,
                "training_dataset": "COCO 2017",
            },
        },
        {
            "name": "YOLOv11s",
            "version": "1.0.0",
            "stage": "staging",
            "model_path": "/models/yolo11s.pt",
            "model_format": "pytorch",
            "metrics": {
                "accuracy": 0.89,
                "latency_ms": 78,
                "map50": 0.82,
                "model_size_mb": 21.5,
            },
            "metadata": {
                "description": "YOLOv11 small model for balanced performance",
                "input_size": [640, 640],
                "classes": 80,
                "training_dataset": "COCO 2017",
            },
        },
        {
            "name": "YOLOv11m",
            "version": "1.0.0",
            "stage": "development",
            "model_path": "/models/yolo11m.pt",
            "model_format": "pytorch",
            "metrics": {
                "accuracy": 0.92,
                "latency_ms": 125,
                "map50": 0.85,
                "model_size_mb": 49.7,
            },
            "metadata": {
                "description": "YOLOv11 medium model for high accuracy",
                "input_size": [640, 640],
                "classes": 80,
                "training_dataset": "COCO 2017",
            },
        },
        {
            "name": "YOLOv11n-TensorRT",
            "version": "1.0.0",
            "stage": "canary",
            "model_path": "/models/yolo11n.trt",
            "model_format": "tensorrt",
            "metrics": {
                "accuracy": 0.85,
                "latency_ms": 15,
                "map50": 0.78,
                "model_size_mb": 8.1,
            },
            "metadata": {
                "description": "TensorRT optimized YOLOv11 nano for GPU inference",
                "input_size": [640, 640],
                "classes": 80,
                "gpu_required": True,
                "tensorrt_version": "8.6.1",
            },
        },
    ]

    model_ids = []
    admin_user_id = await conn.fetchval("SELECT id FROM users WHERE username = 'admin'")

    for model in models_data:
        model_id = await conn.fetchval(
            """
            INSERT INTO model_registry (name, version, stage, model_path, model_format, metrics, metadata, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (name, version) DO NOTHING
            RETURNING id
            """,
            model["name"],
            model["version"],
            model["stage"],
            model["model_path"],
            model["model_format"],
            model["metrics"],
            model["metadata"],
            admin_user_id,
        )
        if model_id:
            model_ids.append(model_id)
            print(
                f"Created model: {model['name']} v{model['version']} (ID: {model_id})"
            )

    return model_ids


async def create_sample_processing_jobs(
    conn: asyncpg.Connection, camera_ids: list[str]
) -> None:
    """Create sample processing jobs."""
    job_types = ["inference", "training", "validation"]
    statuses = ["completed", "processing", "pending", "failed"]

    import random
    from datetime import datetime, timedelta

    for _i in range(20):
        camera_id = random.choice(camera_ids)
        job_type = random.choice(job_types)
        status = random.choice(statuses)

        # Create realistic timestamps
        created_at = datetime.now() - timedelta(hours=random.randint(1, 72))
        started_at = (
            created_at + timedelta(minutes=random.randint(1, 30))
            if status != "pending"
            else None
        )
        completed_at = (
            started_at + timedelta(minutes=random.randint(1, 60))
            if status == "completed"
            else None
        )

        input_data = {
            "batch_size": random.randint(1, 8),
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
        }

        output_data = None
        if status == "completed":
            output_data = {
                "processed_frames": random.randint(100, 1000),
                "detections_count": random.randint(50, 500),
                "average_confidence": round(random.uniform(0.6, 0.95), 3),
                "processing_time_ms": random.randint(1000, 10000),
            }

        error_message = None
        if status == "failed":
            error_message = random.choice(
                [
                    "CUDA out of memory",
                    "Network timeout",
                    "Invalid input format",
                    "Model loading failed",
                ]
            )

        await conn.execute(
            """
            INSERT INTO processing_jobs (
                camera_id, job_type, status, input_data, output_data,
                error_message, started_at, completed_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            camera_id,
            job_type,
            status,
            input_data,
            output_data,
            error_message,
            started_at,
            completed_at,
            created_at,
        )

    print("Created 20 sample processing jobs")


async def create_sample_inference_results(
    conn: asyncpg.Connection, camera_ids: list[str], model_ids: list[str]
) -> None:
    """Create sample inference results."""
    import random
    from datetime import datetime, timedelta

    # Common vehicle classes in COCO dataset
    vehicle_classes = [
        {"class_id": 2, "name": "car"},
        {"class_id": 3, "name": "motorcycle"},
        {"class_id": 5, "name": "bus"},
        {"class_id": 7, "name": "truck"},
    ]

    for _i in range(100):
        camera_id = random.choice(camera_ids)
        model_id = random.choice(model_ids)

        # Create realistic timestamps (last 24 hours)
        frame_timestamp = datetime.now() - timedelta(
            hours=random.randint(0, 24),
            minutes=random.randint(0, 60),
            seconds=random.randint(0, 60),
        )

        # Generate realistic detections
        num_detections = random.randint(0, 8)
        detections = []

        for _ in range(num_detections):
            vehicle = random.choice(vehicle_classes)
            detection = {
                "class_id": vehicle["class_id"],
                "class_name": vehicle["name"],
                "confidence": round(random.uniform(0.5, 0.98), 3),
                "bbox": {
                    "x1": random.randint(0, 400),
                    "y1": random.randint(0, 300),
                    "x2": random.randint(400, 640),
                    "y2": random.randint(300, 480),
                },
                "track_id": random.randint(1, 1000) if random.random() > 0.3 else None,
            }
            detections.append(detection)

        metrics = {
            "inference_time_ms": random.randint(15, 150),
            "preprocessing_time_ms": random.randint(1, 10),
            "postprocessing_time_ms": random.randint(1, 5),
            "total_objects": num_detections,
            "gpu_memory_mb": random.randint(500, 2000),
        }

        processing_time_ms = (
            metrics["inference_time_ms"]
            + metrics["preprocessing_time_ms"]
            + metrics["postprocessing_time_ms"]
        )

        image_path = (
            f"/data/frames/{camera_id}/{frame_timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        await conn.execute(
            """
            INSERT INTO inference_results (
                camera_id, model_id, frame_timestamp, detections,
                metrics, image_path, processing_time_ms
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            camera_id,
            model_id,
            frame_timestamp,
            detections,
            metrics,
            image_path,
            processing_time_ms,
        )

    print("Created 100 sample inference results")


async def create_sample_audit_logs(conn: asyncpg.Connection) -> None:
    """Create sample audit log entries."""
    import random
    from datetime import datetime, timedelta

    actions = [
        "login",
        "logout",
        "create_camera",
        "update_camera",
        "delete_camera",
        "deploy_model",
        "inference_request",
        "config_update",
        "user_create",
    ]

    resource_types = ["user", "camera", "model", "config", "system"]

    ip_addresses = [
        "192.168.1.100",
        "192.168.1.101",
        "192.168.1.102",
        "10.0.1.50",
        "10.0.1.51",
        "172.16.0.100",
    ]

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "ITS-Camera-AI-CLI/1.0.0",
        "Python/3.12 aiohttp/3.8.4",
    ]

    admin_user_id = await conn.fetchval("SELECT id FROM users WHERE username = 'admin'")

    for _i in range(50):
        action = random.choice(actions)
        resource_type = random.choice(resource_types)
        ip_address = random.choice(ip_addresses)
        user_agent = random.choice(user_agents)

        created_at = datetime.now() - timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 24),
            minutes=random.randint(0, 60),
        )

        details = {
            "action": action,
            "success": random.choice([True, True, True, False]),  # 75% success rate
            "duration_ms": random.randint(10, 1000),
            "additional_data": f"Sample {resource_type} operation",
        }

        await conn.execute(
            """
            INSERT INTO audit_logs (
                user_id, action, resource_type, details,
                ip_address, user_agent, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            admin_user_id,
            action,
            resource_type,
            details,
            ip_address,
            user_agent,
            created_at,
        )

    print("Created 50 sample audit log entries")


async def main() -> None:
    """Main seeding function."""
    settings = Settings()

    print("🌱 Starting database seeding for ITS Camera AI...")

    try:
        # Parse database URL
        import urllib.parse as urlparse

        parsed_url = urlparse.urlparse(settings.database_url)

        conn = await asyncpg.connect(
            host=parsed_url.hostname,
            port=parsed_url.port or 5432,
            user=parsed_url.username,
            password=parsed_url.password,
            database=parsed_url.path[1:],  # Remove leading slash
        )

        print("✅ Connected to database")

        # Create sample data
        print("\n📷 Creating sample cameras...")
        camera_ids = await create_sample_cameras(conn)

        print("\n🤖 Creating sample models...")
        model_ids = await create_sample_models(conn)

        print("\n⚡ Creating sample processing jobs...")
        await create_sample_processing_jobs(conn, camera_ids)

        print("\n🎯 Creating sample inference results...")
        await create_sample_inference_results(conn, camera_ids, model_ids)

        print("\n📝 Creating sample audit logs...")
        await create_sample_audit_logs(conn)

        await conn.close()

        print("\n✅ Database seeding completed successfully!")
        print("\n📊 Summary:")
        print(f"   • Cameras: {len(camera_ids)}")
        print(f"   • Models: {len(model_ids)}")
        print("   • Processing Jobs: 20")
        print("   • Inference Results: 100")
        print("   • Audit Logs: 50")

    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
