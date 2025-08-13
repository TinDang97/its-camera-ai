"""Add speed limits table

Revision ID: 003
Revises: 002
Create Date: 2025-01-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add speed limits table with indexes for high-performance lookups."""
    # Create speed_limits table
    op.create_table(
        'speed_limits',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        # Zone and location information
        sa.Column('zone_id', sa.String(length=100), nullable=False, comment='Traffic zone identifier'),
        sa.Column('zone_name', sa.String(length=200), nullable=True, comment='Human-readable zone name'),
        sa.Column('location_description', sa.String(length=300), nullable=True, comment='Detailed location description'),
        
        # Vehicle type classification
        sa.Column('vehicle_type', sa.String(length=50), nullable=False, server_default='general', 
                 comment='Vehicle type (general/car/truck/motorcycle/bus/emergency)'),
        
        # Speed limit values (km/h)
        sa.Column('speed_limit_kmh', sa.Float(), nullable=False, comment='Speed limit in kilometers per hour'),
        sa.Column('tolerance_kmh', sa.Float(), nullable=False, server_default='5.0', comment='Tolerance threshold in km/h'),
        
        # Time-based restrictions
        sa.Column('effective_start_time', sa.String(length=8), nullable=True, comment='Daily start time (HH:MM:SS format)'),
        sa.Column('effective_end_time', sa.String(length=8), nullable=True, comment='Daily end time (HH:MM:SS format)'),
        sa.Column('days_of_week', postgresql.JSONB(astext_type=sa.Text()), nullable=True, 
                 comment='Days of week (0=Monday to 6=Sunday), null=all days'),
        
        # Environmental conditions
        sa.Column('weather_conditions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, 
                 comment='Weather conditions when limit applies (null=all conditions)'),
        sa.Column('minimum_visibility', sa.Float(), nullable=True, comment='Minimum visibility in meters for this limit'),
        
        # Enforcement configuration
        sa.Column('enforcement_enabled', sa.Boolean(), nullable=False, server_default='true', 
                 comment='Whether to enforce this speed limit'),
        sa.Column('warning_threshold', sa.Float(), nullable=True, comment='Speed threshold for warnings (km/h over limit)'),
        sa.Column('violation_threshold', sa.Float(), nullable=False, server_default='10.0', 
                 comment='Speed threshold for violations (km/h over limit)'),
        
        # Priority and validity
        sa.Column('priority', sa.Integer(), nullable=False, server_default='100', 
                 comment='Priority when multiple limits apply (lower=higher priority)'),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False, comment='Speed limit validity start date'),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True, comment='Speed limit validity end date (null=permanent)'),
        
        # Geographic boundaries (optional)
        sa.Column('geographic_bounds', postgresql.JSONB(astext_type=sa.Text()), nullable=True, 
                 comment='Geographic boundaries for the speed limit zone'),
        
        # Administrative information
        sa.Column('authority', sa.String(length=100), nullable=True, comment='Authority that set this speed limit'),
        sa.Column('regulation_reference', sa.String(length=200), nullable=True, comment='Legal regulation reference'),
        sa.Column('last_updated_by', sa.String(length=100), nullable=True, comment='User who last updated this limit'),
        
        sa.PrimaryKeyConstraint('id'),
        comment='Dynamic speed limits by zone and vehicle type with time-based restrictions'
    )
    
    # Create indexes for performance optimization
    op.create_index('idx_speed_limit_zone_vehicle', 'speed_limits', ['zone_id', 'vehicle_type'])
    op.create_index('idx_speed_limit_zone_active', 'speed_limits', ['zone_id', 'enforcement_enabled'])
    op.create_index('idx_speed_limit_vehicle_type', 'speed_limits', ['vehicle_type'])
    op.create_index('idx_speed_limit_validity', 'speed_limits', ['valid_from', 'valid_until'])
    op.create_index('idx_speed_limit_priority', 'speed_limits', ['priority'])
    op.create_index('idx_speed_limit_lookup', 'speed_limits', ['zone_id', 'vehicle_type', 'enforcement_enabled', 'valid_from'])
    
    # Partial index for active limits (PostgreSQL specific)
    op.execute("""
        CREATE INDEX idx_speed_limit_active 
        ON speed_limits (zone_id, vehicle_type, priority) 
        WHERE enforcement_enabled = true AND (valid_until IS NULL OR valid_until > NOW())
    """)

    # Insert some default speed limits
    op.execute("""
        INSERT INTO speed_limits (
            id, zone_id, zone_name, vehicle_type, speed_limit_kmh, tolerance_kmh,
            enforcement_enabled, violation_threshold, priority, valid_from, authority
        ) VALUES 
        -- Default general speed limits
        (gen_random_uuid(), 'default', 'Default Zone', 'general', 50.0, 5.0, true, 10.0, 100, NOW(), 'System'),
        (gen_random_uuid(), 'default', 'Default Zone', 'truck', 40.0, 5.0, true, 10.0, 100, NOW(), 'System'),
        (gen_random_uuid(), 'default', 'Default Zone', 'bus', 45.0, 5.0, true, 10.0, 100, NOW(), 'System'),
        (gen_random_uuid(), 'default', 'Default Zone', 'motorcycle', 50.0, 5.0, true, 10.0, 100, NOW(), 'System'),
        (gen_random_uuid(), 'default', 'Default Zone', 'emergency', 70.0, 10.0, false, 20.0, 50, NOW(), 'System'),
        
        -- School zone limits (example)
        (gen_random_uuid(), 'school_zone_1', 'School Zone - Main Street', 'general', 30.0, 3.0, true, 5.0, 10, NOW(), 'Traffic Authority'),
        (gen_random_uuid(), 'school_zone_1', 'School Zone - Main Street', 'truck', 25.0, 3.0, true, 5.0, 10, NOW(), 'Traffic Authority'),
        
        -- Highway limits (example)
        (gen_random_uuid(), 'highway_a1', 'Highway A1', 'general', 80.0, 10.0, true, 15.0, 100, NOW(), 'Highway Authority'),
        (gen_random_uuid(), 'highway_a1', 'Highway A1', 'truck', 70.0, 8.0, true, 12.0, 100, NOW(), 'Highway Authority')
    """)


def downgrade() -> None:
    """Remove speed limits table."""
    op.drop_table('speed_limits')