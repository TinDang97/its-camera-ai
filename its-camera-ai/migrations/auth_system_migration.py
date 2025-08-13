"""
Database migration for comprehensive authentication system.

Creates and modifies tables for enterprise authentication features:
- Enhanced User table with MFA and security fields
- Permission table for fine-grained access control
- Updated Role table with permission relationships
- SecurityAuditLog table for compliance logging
- UserSession table for session management
"""

from sqlalchemy import (
    MetaData,
    create_engine,
)
from sqlalchemy.sql import text

from src.its_camera_ai.core.config import get_settings


def get_database_connection():
    """Get database connection for migrations."""
    settings = get_settings()
    # Use sync engine for migrations
    sync_url = settings.get_database_url(async_driver=False)
    return create_engine(sync_url)


def upgrade():
    """Apply authentication system upgrades."""
    engine = get_database_connection()
    MetaData()

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            print("Starting authentication system migration...")

            # 1. Add new columns to existing user table
            print("1. Enhancing user table with security fields...")

            # Add MFA fields
            conn.execute(text("""
                ALTER TABLE "user"
                ADD COLUMN IF NOT EXISTS mfa_enabled BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS mfa_secret VARCHAR(255),
                ADD COLUMN IF NOT EXISTS mfa_backup_codes TEXT
            """))

            # Add security fields
            conn.execute(text("""
                ALTER TABLE "user"
                ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS last_login TIMESTAMP,
                ADD COLUMN IF NOT EXISTS last_password_change TIMESTAMP,
                ADD COLUMN IF NOT EXISTS password_history TEXT,
                ADD COLUMN IF NOT EXISTS account_locked_until TIMESTAMP,
                ADD COLUMN IF NOT EXISTS last_login_ip VARCHAR(45),
                ADD COLUMN IF NOT EXISTS last_login_device VARCHAR(255),
                ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP
            """))

            # Increase password hash column size for better security
            conn.execute(text("""
                ALTER TABLE "user"
                ALTER COLUMN hashed_password TYPE VARCHAR(255)
            """))

            print("✓ User table enhanced successfully")

            # 2. Create permission table
            print("2. Creating permission table...")

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS permission (
                    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    resource VARCHAR(100) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    description VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))

            # Create unique index on resource + action
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_permission_resource_action
                ON permission(resource, action)
            """))

            print("✓ Permission table created successfully")

            # 3. Create role-permission association table
            print("3. Creating role-permission relationship...")

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_id VARCHAR REFERENCES role(id) ON DELETE CASCADE,
                    permission_id VARCHAR REFERENCES permission(id) ON DELETE CASCADE,
                    PRIMARY KEY (role_id, permission_id)
                )
            """))

            print("✓ Role-permission relationship created successfully")

            # 4. Create security audit log table
            print("4. Creating security audit log table...")

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    event_type VARCHAR(50) NOT NULL,
                    user_id VARCHAR REFERENCES "user"(id) ON DELETE SET NULL,
                    username VARCHAR(50),
                    ip_address VARCHAR(45),
                    user_agent VARCHAR(500),
                    session_id VARCHAR(255),
                    resource VARCHAR(100),
                    action VARCHAR(50),
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    details TEXT,
                    risk_score INTEGER DEFAULT 0,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))

            # Create indexes for audit log queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON security_audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON security_audit_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON security_audit_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_log_risk_score ON security_audit_log(risk_score);
                CREATE INDEX IF NOT EXISTS idx_audit_log_success ON security_audit_log(success);
            """))

            print("✓ Security audit log table created successfully")

            # 5. Create user session table
            print("5. Creating user session table...")

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_session (
                    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id VARCHAR NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
                    ip_address VARCHAR(45),
                    user_agent VARCHAR(500),
                    device_fingerprint VARCHAR(255),
                    mfa_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    last_activity TIMESTAMP NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))

            # Create indexes for session queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_session_user_id ON user_session(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_session_expires_at ON user_session(expires_at);
                CREATE INDEX IF NOT EXISTS idx_user_session_is_active ON user_session(is_active);
                CREATE INDEX IF NOT EXISTS idx_user_session_last_activity ON user_session(last_activity);
            """))

            print("✓ User session table created successfully")

            # 6. Insert default permissions
            print("6. Inserting default RBAC permissions...")

            default_permissions = [
                # User management
                ('users:create', 'users', 'create', 'Create new users'),
                ('users:read', 'users', 'read', 'View user information'),
                ('users:update', 'users', 'update', 'Update user information'),
                ('users:delete', 'users', 'delete', 'Delete users'),

                # Role management
                ('roles:create', 'roles', 'create', 'Create new roles'),
                ('roles:read', 'roles', 'read', 'View role information'),
                ('roles:update', 'roles', 'update', 'Update role information'),
                ('roles:delete', 'roles', 'delete', 'Delete roles'),

                # Camera management
                ('cameras:create', 'cameras', 'create', 'Create new camera connections'),
                ('cameras:read', 'cameras', 'read', 'View camera feeds and information'),
                ('cameras:update', 'cameras', 'update', 'Update camera settings'),
                ('cameras:delete', 'cameras', 'delete', 'Remove camera connections'),
                ('cameras:control', 'cameras', 'control', 'Control camera operations (PTZ, recording)'),

                # Analytics
                ('analytics:create', 'analytics', 'create', 'Create analytics configurations'),
                ('analytics:read', 'analytics', 'read', 'View analytics data and reports'),
                ('analytics:update', 'analytics', 'update', 'Update analytics settings'),
                ('analytics:delete', 'analytics', 'delete', 'Delete analytics configurations'),
                ('analytics:manage', 'analytics', 'manage', 'Full analytics management'),
                ('analytics:export', 'analytics', 'export', 'Export analytics data'),

                # Incident management
                ('incidents:create', 'incidents', 'create', 'Create incident reports'),
                ('incidents:read', 'incidents', 'read', 'View incident information'),
                ('incidents:update', 'incidents', 'update', 'Update incident details'),
                ('incidents:delete', 'incidents', 'delete', 'Delete incident reports'),
                ('incidents:manage', 'incidents', 'manage', 'Full incident management'),

                # System management
                ('system:configure', 'system', 'configure', 'Configure system settings'),
                ('system:monitor', 'system', 'monitor', 'Monitor system health'),
                ('system:backup', 'system', 'backup', 'Perform system backups'),
                ('system:restore', 'system', 'restore', 'Restore from backups'),

                # Security and audit
                ('security:audit', 'security', 'audit', 'View security audit logs'),
                ('security:manage', 'security', 'manage', 'Manage security settings'),
                ('logs:read', 'logs', 'read', 'View system logs'),
                ('logs:export', 'logs', 'export', 'Export log data'),

                # Reports
                ('reports:create', 'reports', 'create', 'Create reports'),
                ('reports:read', 'reports', 'read', 'View reports'),
                ('reports:update', 'reports', 'update', 'Update reports'),
                ('reports:delete', 'reports', 'delete', 'Delete reports'),

                # Public access
                ('public:read', 'public', 'read', 'Access public information'),
            ]

            for name, resource, action, description in default_permissions:
                conn.execute(text("""
                    INSERT INTO permission (name, resource, action, description)
                    VALUES (:name, :resource, :action, :description)
                    ON CONFLICT (name) DO NOTHING
                """), {
                    'name': name,
                    'resource': resource,
                    'action': action,
                    'description': description
                })

            print(f"✓ Inserted {len(default_permissions)} default permissions")

            # 7. Create default roles and assign permissions
            print("7. Creating default roles with permissions...")

            # Create roles if they don't exist
            default_roles = [
                ('admin', 'System Administrator with full access'),
                ('operator', 'System Operator with operational access'),
                ('analyst', 'Data Analyst with analytics access'),
                ('viewer', 'Read-only access to cameras and analytics'),
                ('auditor', 'Security auditor with audit access'),
                ('guest', 'Limited guest access'),
            ]

            for name, description in default_roles:
                conn.execute(text("""
                    INSERT INTO role (name, description)
                    VALUES (:name, :description)
                    ON CONFLICT (name) DO NOTHING
                """), {'name': name, 'description': description})

            # Assign permissions to roles
            role_permissions_map = {
                'admin': [  # Full access
                    'users:create', 'users:read', 'users:update', 'users:delete',
                    'roles:create', 'roles:read', 'roles:update', 'roles:delete',
                    'cameras:create', 'cameras:read', 'cameras:update', 'cameras:delete', 'cameras:control',
                    'analytics:create', 'analytics:read', 'analytics:update', 'analytics:delete', 'analytics:manage', 'analytics:export',
                    'incidents:create', 'incidents:read', 'incidents:update', 'incidents:delete', 'incidents:manage',
                    'system:configure', 'system:monitor', 'system:backup', 'system:restore',
                    'security:audit', 'security:manage', 'logs:read', 'logs:export',
                    'reports:create', 'reports:read', 'reports:update', 'reports:delete'
                ],
                'operator': [  # Operational access
                    'cameras:read', 'cameras:update', 'cameras:control',
                    'analytics:read', 'analytics:create', 'incidents:manage',
                    'users:read', 'system:monitor'
                ],
                'analyst': [  # Analytics focus
                    'analytics:read', 'analytics:create', 'analytics:export',
                    'cameras:read', 'incidents:read', 'reports:create', 'reports:read'
                ],
                'viewer': [  # Read-only access
                    'cameras:read', 'analytics:read', 'incidents:read', 'reports:read'
                ],
                'auditor': [  # Security and audit access
                    'security:audit', 'logs:read', 'users:read', 'roles:read',
                    'analytics:read', 'reports:read'
                ],
                'guest': [  # Minimal access
                    'cameras:read', 'public:read'
                ]
            }

            for role_name, permission_names in role_permissions_map.items():
                for permission_name in permission_names:
                    conn.execute(text("""
                        INSERT INTO role_permissions (role_id, permission_id)
                        SELECT r.id, p.id
                        FROM role r, permission p
                        WHERE r.name = :role_name AND p.name = :permission_name
                        ON CONFLICT DO NOTHING
                    """), {
                        'role_name': role_name,
                        'permission_name': permission_name
                    })

            print("✓ Default roles and permissions configured successfully")

            # 8. Create indexes for performance
            print("8. Creating performance indexes...")

            # User table indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_username ON "user"(username);
                CREATE INDEX IF NOT EXISTS idx_user_email ON "user"(email);
                CREATE INDEX IF NOT EXISTS idx_user_is_active ON "user"(is_active);
                CREATE INDEX IF NOT EXISTS idx_user_last_login ON "user"(last_login);
                CREATE INDEX IF NOT EXISTS idx_user_mfa_enabled ON "user"(mfa_enabled);
            """))

            # Role table indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_role_name ON role(name);
            """))

            print("✓ Performance indexes created successfully")

            # 9. Create triggers for automatic timestamp updates
            print("9. Creating timestamp update triggers...")

            # Updated_at trigger function
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))

            # Apply triggers to tables
            tables_with_updated_at = [
                '"user"', 'role', 'permission',
                'security_audit_log', 'user_session'
            ]

            for table in tables_with_updated_at:
                conn.execute(text(f"""
                    DROP TRIGGER IF EXISTS update_{table.strip('"')}_updated_at ON {table};
                    CREATE TRIGGER update_{table.strip('"')}_updated_at
                        BEFORE UPDATE ON {table}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                """))

            print("✓ Timestamp update triggers created successfully")

            # 10. Create stored procedures for common operations
            print("10. Creating authentication stored procedures...")

            # Procedure for session cleanup
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
                RETURNS INTEGER AS $$
                DECLARE
                    deleted_count INTEGER;
                BEGIN
                    DELETE FROM user_session
                    WHERE expires_at < NOW() OR is_active = FALSE;

                    GET DIAGNOSTICS deleted_count = ROW_COUNT;
                    RETURN deleted_count;
                END;
                $$ LANGUAGE plpgsql;
            """))

            # Procedure for audit log cleanup
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(retention_days INTEGER DEFAULT 365)
                RETURNS INTEGER AS $$
                DECLARE
                    deleted_count INTEGER;
                BEGIN
                    DELETE FROM security_audit_log
                    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;

                    GET DIAGNOSTICS deleted_count = ROW_COUNT;
                    RETURN deleted_count;
                END;
                $$ LANGUAGE plpgsql;
            """))

            print("✓ Stored procedures created successfully")

            # Commit all changes
            trans.commit()

            print("🎉 Authentication system migration completed successfully!")
            print("\nSummary of changes:")
            print("- Enhanced user table with MFA and security fields")
            print("- Created permission table with fine-grained access control")
            print("- Added role-permission relationship table")
            print("- Created security audit log table for compliance")
            print("- Added user session management table")
            print("- Inserted default RBAC permissions and roles")
            print("- Created performance indexes and triggers")
            print("- Added maintenance stored procedures")

        except Exception as e:
            trans.rollback()
            print(f"❌ Migration failed: {str(e)}")
            raise


def downgrade():
    """Rollback authentication system changes."""
    engine = get_database_connection()

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            print("Rolling back authentication system migration...")

            # Drop stored procedures
            conn.execute(text("DROP FUNCTION IF EXISTS cleanup_expired_sessions()"))
            conn.execute(text("DROP FUNCTION IF EXISTS cleanup_old_audit_logs(INTEGER)"))
            conn.execute(text("DROP FUNCTION IF EXISTS update_updated_at_column()"))

            # Drop tables in reverse dependency order
            conn.execute(text("DROP TABLE IF EXISTS user_session CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS security_audit_log CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS role_permissions CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS permission CASCADE"))

            # Remove added columns from user table
            user_columns_to_drop = [
                'mfa_enabled', 'mfa_secret', 'mfa_backup_codes',
                'failed_login_attempts', 'last_login', 'last_password_change',
                'password_history', 'account_locked_until', 'last_login_ip',
                'last_login_device', 'email_verified_at'
            ]

            for column in user_columns_to_drop:
                try:
                    conn.execute(text(f'ALTER TABLE "user" DROP COLUMN IF EXISTS {column}'))
                except Exception:
                    pass  # Column might not exist

            # Restore original password hash column size
            conn.execute(text('''
                ALTER TABLE "user"
                ALTER COLUMN hashed_password TYPE VARCHAR(128)
            '''))

            trans.commit()
            print("✅ Authentication system migration rolled back successfully")

        except Exception as e:
            trans.rollback()
            print(f"❌ Rollback failed: {str(e)}")
            raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade()
