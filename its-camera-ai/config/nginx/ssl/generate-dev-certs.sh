#!/bin/bash

# Generate self-signed SSL certificates for development
# ITS Camera AI Project

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CERT_FILE="${SCRIPT_DIR}/cert.pem"
KEY_FILE="${SCRIPT_DIR}/key.pem"

echo "🔐 Generating self-signed SSL certificates for development..."

# Check if certificates already exist
if [[ -f "$CERT_FILE" && -f "$KEY_FILE" ]]; then
    echo "⚠️  SSL certificates already exist. Overwrite? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Certificate generation cancelled."
        exit 0
    fi
fi

# Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -subj "/C=US/ST=CA/L=San Francisco/O=ITS Camera AI/OU=Development/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:*.localhost,DNS:its-camera-ai.local,DNS:*.its-camera-ai.local,DNS:dev.its-camera-ai.local,IP:127.0.0.1,IP:::1"

# Set proper permissions
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo "✅ SSL certificates generated successfully!"
echo "📁 Certificate: $CERT_FILE"
echo "🔑 Private Key: $KEY_FILE"
echo ""
echo "🚀 To use these certificates:"
echo "   1. Add 'its-camera-ai.local' to your /etc/hosts file:"
echo "      echo '127.0.0.1 its-camera-ai.local dev.its-camera-ai.local' | sudo tee -a /etc/hosts"
echo "   2. Start the application with docker-compose"
echo "   3. Access via https://its-camera-ai.local"
echo ""
echo "⚠️  Note: Your browser will show a security warning for self-signed certificates."
echo "   Click 'Advanced' and 'Proceed to localhost (unsafe)' to continue."