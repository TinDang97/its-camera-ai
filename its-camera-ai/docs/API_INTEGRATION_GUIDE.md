# ITS Camera AI - API Integration Guide

**Version:** 2.0  
**Date:** January 16, 2025  
**Target Audience:** System Integrators, Traffic Authorities, Third-party Developers  

## Overview

This guide provides comprehensive documentation for integrating with the ITS Camera AI traffic monitoring system. The system offers RESTful APIs, gRPC services, and real-time streaming capabilities designed for traffic management applications.

## Quick Start

### Authentication
All API requests require JWT authentication. Obtain your API key from the customer portal.

```bash
# Get authentication token
curl -X POST "https://api.its-camera-ai.com/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

### Base URLs
- **Production**: `https://api.its-camera-ai.com`
- **Staging**: `https://staging-api.its-camera-ai.com`
- **API Version**: `v1` (current)

## Core API Endpoints

### Camera Management

#### Register Camera
```http
POST /api/v1/cameras
Content-Type: application/json
Authorization: Bearer {token}

{
  "camera_id": "cam_001",
  "name": "Main Street Intersection",
  "stream_url": "rtsp://camera.example.com/stream1",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "address": "Main St & 1st Ave"
  },
  "configuration": {
    "resolution": [1920, 1080],
    "fps": 30,
    "protocol": "RTSP"
  }
}
```

#### Get Camera Status
```http
GET /api/v1/cameras/{camera_id}
Authorization: Bearer {token}
```

Response:
```json
{
  "camera_id": "cam_001",
  "name": "Main Street Intersection",
  "status": "active",
  "health": {
    "connection": "healthy",
    "stream_quality": "good",
    "last_frame": "2025-01-16T10:30:00Z"
  },
  "metrics": {
    "fps": 29.8,
    "detection_count": 156,
    "processing_latency_ms": 75
  }
}
```

### Traffic Analytics

#### Get Real-time Traffic Metrics
```http
GET /api/v1/analytics/traffic-metrics
Authorization: Bearer {token}
Query Parameters:
  - camera_id: string (optional)
  - start_time: ISO8601 datetime
  - end_time: ISO8601 datetime
  - aggregation: "5min" | "hourly" | "daily"
```

Response:
```json
{
  "camera_id": "cam_001",
  "time_period": {
    "start": "2025-01-16T10:00:00Z",
    "end": "2025-01-16T11:00:00Z"
  },
  "metrics": {
    "vehicle_count": 245,
    "average_speed": 35.2,
    "traffic_density": 0.68,
    "congestion_level": "moderate",
    "vehicle_types": {
      "car": 198,
      "truck": 24,
      "motorcycle": 15,
      "bus": 8
    }
  }
}
```

#### Get Vehicle Detections
```http
GET /api/v1/analytics/detections
Authorization: Bearer {token}
Query Parameters:
  - camera_id: string (required)
  - start_time: ISO8601 datetime
  - end_time: ISO8601 datetime
  - vehicle_type: string (optional)
  - confidence_threshold: float (0.0-1.0)
```

### Incident Management

#### Get Traffic Violations
```http
GET /api/v1/incidents/violations
Authorization: Bearer {token}
Query Parameters:
  - camera_id: string (optional)
  - violation_type: "speed" | "lane" | "red_light" | "wrong_way"
  - severity: "low" | "medium" | "high"
  - start_time: ISO8601 datetime
  - end_time: ISO8601 datetime
```

Response:
```json
{
  "violations": [
    {
      "violation_id": "viol_12345",
      "camera_id": "cam_001",
      "violation_type": "speed",
      "severity": "high",
      "timestamp": "2025-01-16T10:15:30Z",
      "details": {
        "speed_limit": 35,
        "detected_speed": 58,
        "vehicle_type": "car",
        "license_plate": "ABC123",
        "confidence": 0.95
      },
      "location": {
        "latitude": 40.7128,
        "longitude": -74.0060
      }
    }
  ]
}
```

## Real-time Streaming

### Server-Sent Events (SSE)
Subscribe to real-time traffic events:

```javascript
const eventSource = new EventSource(
  'https://api.its-camera-ai.com/api/v1/stream/events?camera_id=cam_001',
  {
    headers: {
      'Authorization': 'Bearer your_token'
    }
  }
);

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Traffic event:', data);
};

// Event types: 'detection', 'violation', 'incident', 'alert'
eventSource.addEventListener('violation', function(event) {
  const violation = JSON.parse(event.data);
  handleTrafficViolation(violation);
});
```

### WebSocket Connection
For bidirectional communication:

```javascript
const ws = new WebSocket('wss://api.its-camera-ai.com/api/v1/ws/traffic');

ws.onopen = function() {
  // Subscribe to specific camera feeds
  ws.send(JSON.stringify({
    action: 'subscribe',
    camera_ids: ['cam_001', 'cam_002'],
    event_types: ['detection', 'violation']
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  handleRealTimeEvent(data);
};
```

## Legacy System Integration

### Traffic Management System (TMS) Integration

#### NTCIP Protocol Support
```http
POST /api/v1/integrations/ntcip/traffic-data
Content-Type: application/json
Authorization: Bearer {token}

{
  "intersection_id": "int_001",
  "traffic_data": {
    "volume": 245,
    "occupancy": 0.68,
    "speed": 35.2,
    "timestamp": "2025-01-16T10:30:00Z"
  },
  "signal_timing": {
    "cycle_length": 120,
    "green_time": 45,
    "yellow_time": 3,
    "red_time": 72
  }
}
```

#### SCATS Integration
```http
POST /api/v1/integrations/scats/detector-data
Content-Type: application/json
Authorization: Bearer {token}

{
  "detector_id": "det_001",
  "site_number": 1001,
  "data": {
    "flow": 245,
    "occupancy": 68,
    "gap": 2.5,
    "headway": 3.2
  }
}
```

### Emergency Services Integration

#### Alert Forwarding
```http
POST /api/v1/integrations/emergency/alert
Content-Type: application/json
Authorization: Bearer {token}

{
  "alert_id": "alert_12345",
  "incident_type": "accident",
  "severity": "high",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "address": "Main St & 1st Ave"
  },
  "description": "Multi-vehicle accident blocking intersection",
  "emergency_contacts": ["911", "traffic_control"],
  "evidence": {
    "video_url": "https://storage.its-camera-ai.com/evidence/video_12345.mp4",
    "images": ["image_1.jpg", "image_2.jpg"]
  }
}
```

## gRPC Services

### High-Performance Streaming
For low-latency, high-throughput applications:

```protobuf
// traffic_monitoring.proto
service TrafficMonitoringService {
  rpc GetRealTimeDetections(DetectionRequest) returns (stream Detection);
  rpc ProcessVideoStream(stream VideoFrame) returns (stream ProcessingResult);
  rpc GetTrafficMetrics(MetricsRequest) returns (TrafficMetrics);
}

message Detection {
  string detection_id = 1;
  string camera_id = 2;
  int64 timestamp = 3;
  BoundingBox bbox = 4;
  string vehicle_type = 5;
  float confidence = 6;
  float speed = 7;
  string license_plate = 8;
}
```

### Client Examples

#### Python gRPC Client
```python
import grpc
import traffic_monitoring_pb2
import traffic_monitoring_pb2_grpc

# Create gRPC channel
channel = grpc.insecure_channel('api.its-camera-ai.com:9090')
stub = traffic_monitoring_pb2_grpc.TrafficMonitoringServiceStub(channel)

# Stream real-time detections
request = traffic_monitoring_pb2.DetectionRequest(
    camera_id="cam_001",
    confidence_threshold=0.8
)

for detection in stub.GetRealTimeDetections(request):
    print(f"Vehicle detected: {detection.vehicle_type} at {detection.timestamp}")
```

#### Go gRPC Client
```go
package main

import (
    "context"
    "io"
    "log"
    
    "google.golang.org/grpc"
    pb "path/to/traffic_monitoring"
)

func main() {
    conn, err := grpc.Dial("api.its-camera-ai.com:9090", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    
    client := pb.NewTrafficMonitoringServiceClient(conn)
    
    stream, err := client.GetRealTimeDetections(context.Background(), &pb.DetectionRequest{
        CameraId: "cam_001",
        ConfidenceThreshold: 0.8,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    for {
        detection, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }
        
        log.Printf("Detection: %s at %s", detection.VehicleType, detection.Timestamp)
    }
}
```

## SDK Libraries

### JavaScript/Node.js SDK
```bash
npm install @its-camera-ai/sdk
```

```javascript
const { ITSCameraAI } = require('@its-camera-ai/sdk');

const client = new ITSCameraAI({
  apiKey: 'your_api_key',
  baseURL: 'https://api.its-camera-ai.com'
});

// Get camera status
const camera = await client.cameras.get('cam_001');
console.log('Camera status:', camera.status);

// Stream real-time events
client.events.subscribe('cam_001', {
  onDetection: (detection) => console.log('Detection:', detection),
  onViolation: (violation) => console.log('Violation:', violation)
});
```

### Python SDK
```bash
pip install its-camera-ai-sdk
```

```python
from its_camera_ai import ITSCameraAI

client = ITSCameraAI(api_key='your_api_key')

# Get traffic metrics
metrics = client.analytics.get_traffic_metrics(
    camera_id='cam_001',
    start_time='2025-01-16T10:00:00Z',
    end_time='2025-01-16T11:00:00Z'
)

print(f"Vehicle count: {metrics.vehicle_count}")
print(f"Average speed: {metrics.average_speed}")
```

## Webhook Integration

### Configure Webhooks
```http
POST /api/v1/webhooks
Content-Type: application/json
Authorization: Bearer {token}

{
  "url": "https://your-system.com/webhook/traffic-events",
  "events": ["violation", "incident", "alert"],
  "secret": "your_webhook_secret",
  "active": true
}
```

### Webhook Payload Example
```json
{
  "event_type": "violation",
  "timestamp": "2025-01-16T10:15:30Z",
  "webhook_id": "wh_12345",
  "data": {
    "violation_id": "viol_12345",
    "camera_id": "cam_001",
    "violation_type": "speed",
    "severity": "high",
    "details": {
      "speed_limit": 35,
      "detected_speed": 58,
      "vehicle_type": "car",
      "license_plate": "ABC123"
    }
  }
}
```

## Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

### Error Response Format
```json
{
  "error": {
    "code": "CAMERA_NOT_FOUND",
    "message": "Camera with ID 'cam_001' was not found",
    "details": {
      "camera_id": "cam_001",
      "suggested_action": "Verify camera_id or register camera first"
    },
    "timestamp": "2025-01-16T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

## Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|--------|
| Authentication | 10 requests | 1 minute |
| Camera Management | 100 requests | 1 minute |
| Analytics | 1000 requests | 1 minute |
| Real-time Streaming | Unlimited | - |
| Webhooks | 50 requests | 1 minute |

## Testing & Development

### Sandbox Environment
Use the staging environment for development and testing:
- **Base URL**: `https://staging-api.its-camera-ai.com`
- **Test Data**: Pre-configured test cameras and simulated traffic data
- **No Rate Limits**: Unlimited requests for development

### API Explorer
Interactive API documentation available at:
- **Production**: `https://api.its-camera-ai.com/docs`
- **Staging**: `https://staging-api.its-camera-ai.com/docs`

## Support and Resources

### Documentation
- **API Reference**: `https://docs.its-camera-ai.com/api`
- **Integration Examples**: `https://github.com/its-camera-ai/examples`
- **SDK Documentation**: `https://docs.its-camera-ai.com/sdks`

### Support Channels
- **Email**: support@its-camera-ai.com
- **Technical Support**: Available 24/7 for Enterprise customers
- **Community Forum**: `https://community.its-camera-ai.com`

### Getting Help
For integration assistance, please include:
- Your API key (first 8 characters only)
- Request/response examples
- Error messages and status codes
- Your implementation language and SDK version

---

**Last Updated**: January 16, 2025  
**API Version**: v1  
**Document Version**: 2.0