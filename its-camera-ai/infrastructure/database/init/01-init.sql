-- PostgreSQL initialization script for ITS Camera AI
-- This script sets up the database schema and initial configurations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create enum types
CREATE TYPE user_role AS ENUM ('admin', 'operator', 'viewer', 'api');
CREATE TYPE security_level AS ENUM ('public', 'internal', 'confidential', 'restricted');
CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
CREATE TYPE model_stage AS ENUM ('development', 'staging', 'canary', 'production', 'deprecated');

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    location JSONB NOT NULL, -- Store lat, lng, address
    stream_url TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    security_level security_level DEFAULT 'internal',
    configuration JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create processing_jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- 'inference', 'training', 'validation'
    status processing_status DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create model_registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    stage model_stage DEFAULT 'development',
    model_path TEXT NOT NULL,
    model_format VARCHAR(20) DEFAULT 'pytorch', -- 'pytorch', 'onnx', 'tensorrt', 'coreml'
    metrics JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

-- Create inference_results table
CREATE TABLE IF NOT EXISTS inference_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    model_id UUID REFERENCES model_registry(id),
    frame_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    detections JSONB NOT NULL, -- Array of detection objects
    metrics JSONB DEFAULT '{}', -- Performance metrics
    image_path TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_cameras_location ON cameras USING GIN(location);
CREATE INDEX idx_cameras_active ON cameras(is_active);
CREATE INDEX idx_processing_jobs_camera_status ON processing_jobs(camera_id, status);
CREATE INDEX idx_processing_jobs_created_at ON processing_jobs(created_at);
CREATE INDEX idx_model_registry_name_version ON model_registry(name, version);
CREATE INDEX idx_model_registry_stage_active ON model_registry(stage, is_active);
CREATE INDEX idx_inference_results_camera_timestamp ON inference_results(camera_id, frame_timestamp);
CREATE INDEX idx_inference_results_model_id ON inference_results(model_id);
CREATE INDEX idx_audit_logs_user_action ON audit_logs(user_id, action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cameras_updated_at BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, role) 
VALUES (
    'admin',
    'admin@its-camera-ai.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyPTz7fTRWsN7K',
    'admin'
) ON CONFLICT (username) DO NOTHING;

-- Create sample camera entries
INSERT INTO cameras (name, location, stream_url) VALUES 
    ('Main Entrance', '{"lat": 37.7749, "lng": -122.4194, "address": "123 Main St, San Francisco, CA"}', 'rtsp://demo:demo@192.168.1.100:554/stream'),
    ('Parking Lot A', '{"lat": 37.7849, "lng": -122.4094, "address": "456 Oak Ave, San Francisco, CA"}', 'rtsp://demo:demo@192.168.1.101:554/stream'),
    ('Traffic Junction', '{"lat": 37.7649, "lng": -122.4294, "address": "789 Pine St, San Francisco, CA"}', 'rtsp://demo:demo@192.168.1.102:554/stream')
ON CONFLICT DO NOTHING;

-- Create sample model registry entry
INSERT INTO model_registry (name, version, model_path, metrics, metadata) VALUES 
    ('YOLOv11n', '1.0.0', '/models/yolo11n.pt', 
     '{"accuracy": 0.85, "latency_ms": 45, "map50": 0.78}',
     '{"description": "YOLOv11 nano model for real-time inference", "input_size": [640, 640], "classes": 80}')
ON CONFLICT (name, version) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA public TO its_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO its_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO its_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO its_user;

-- Enable row level security (optional, can be configured later)
-- ALTER TABLE inference_results ENABLE ROW LEVEL SECURITY;

COMMIT;