# SSL Certificates for ITS Camera AI

This directory should contain SSL certificates for HTTPS configuration.

## Development Setup

For development, you can generate self-signed certificates:

```bash
# Create self-signed certificate for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem \
    -out cert.pem \
    -subj "/C=US/ST=CA/L=San Francisco/O=ITS Camera AI/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:*.localhost,DNS:its-camera-ai.local,DNS:*.its-camera-ai.local,IP:127.0.0.1"
```

## Production Setup

For production, use certificates from a trusted CA like Let's Encrypt:

```bash
# Using certbot for Let's Encrypt
certbot certonly --nginx -d your-domain.com -d www.your-domain.com

# Copy certificates to this directory
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem key.pem
```

## Required Files

- `cert.pem` - SSL certificate chain
- `key.pem` - Private key

## Security Notes

- Never commit actual SSL certificates to version control
- Use proper file permissions (600) for private keys
- Rotate certificates regularly
- Use strong encryption algorithms (RSA 2048+ or ECDSA 256+)