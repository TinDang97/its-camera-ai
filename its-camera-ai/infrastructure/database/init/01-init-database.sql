-- Database initialization script for ITS Camera AI
-- This script sets up the initial database structure and users

-- Create application database
CREATE DATABASE its_camera_ai;

-- Create users for different environments
CREATE USER its_user WITH ENCRYPTED PASSWORD 'its_password';
CREATE USER its_readonly WITH ENCRYPTED PASSWORD 'readonly_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE its_camera_ai TO its_user;
GRANT CONNECT ON DATABASE its_camera_ai TO its_readonly;

-- Switch to the application database
\c its_camera_ai;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "postgis";  -- For geospatial data if needed
CREATE EXTENSION IF NOT EXISTS "timescaledb"; -- For time series data (if TimescaleDB is available)

-- Grant usage on extensions
GRANT USAGE ON SCHEMA public TO its_user;
GRANT USAGE ON SCHEMA public TO its_readonly;

-- Grant permissions on future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO its_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO its_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO its_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO its_readonly;

-- Create initial schemas for better organization
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS traffic_data;
CREATE SCHEMA IF NOT EXISTS user_management;
CREATE SCHEMA IF NOT EXISTS audit_logs;

-- Grant schema permissions
GRANT USAGE ON SCHEMA ml_models TO its_user;
GRANT USAGE ON SCHEMA traffic_data TO its_user;
GRANT USAGE ON SCHEMA user_management TO its_user;
GRANT USAGE ON SCHEMA audit_logs TO its_user;

GRANT USAGE ON SCHEMA ml_models TO its_readonly;
GRANT USAGE ON SCHEMA traffic_data TO its_readonly;
GRANT USAGE ON SCHEMA user_management TO its_readonly;
GRANT USAGE ON SCHEMA audit_logs TO its_readonly;