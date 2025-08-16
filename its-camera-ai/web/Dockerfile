# Multi-stage Dockerfile for ITS Camera AI Next.js Frontend
# Optimized for development, build, and production stages with security best practices

# Base image with Node.js LTS
FROM node:20-alpine AS base

# Install system dependencies for building native modules
RUN apk add --no-cache \
    libc6-compat \
    curl \
    bash \
    git \
    python3 \
    make \
    g++ \
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 --ingroup nodejs nextjs

# Install dependencies stage
FROM base AS deps

# Copy package files
COPY package.json yarn.lock* ./

# Install all dependencies (including devDependencies for build stage)
RUN yarn install --frozen-lockfile --network-timeout 1000000

# Development stage
FROM base AS development

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source code
COPY --chown=nextjs:nodejs . .

# Expose development port
EXPOSE 3000

# Create .next directory with proper permissions
RUN mkdir -p .next && chown -R nextjs:nodejs .next

# Health check for development
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Switch to non-root user
USER nextjs

# Start development server with hot reload
CMD ["yarn", "dev", "--hostname", "0.0.0.0", "--port", "3000"]

# Build stage
FROM base AS builder

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source code
COPY --chown=nextjs:nodejs . .

# Set production environment for build
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Switch to nextjs user for build
USER nextjs

# Build the application
RUN yarn build

# Production dependencies stage
FROM base AS production-deps

# Copy package files
COPY package.json yarn.lock* ./

# Install only production dependencies
RUN yarn install --frozen-lockfile --production --network-timeout 1000000 \
    && yarn cache clean

# Production stage
FROM base AS production

# Set production environment
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Copy production dependencies
COPY --from=production-deps /app/node_modules ./node_modules

# Copy built application from builder stage
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

# Expose production port
EXPOSE 3000

# Health check for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Switch to non-root user
USER nextjs

# Start production server
CMD ["node", "server.js"]

# Testing stage (extends base for full testing capabilities)
FROM base AS testing

# Install additional testing dependencies
RUN apk add --no-cache \
    chromium \
    nss \
    freetype \
    freetype-dev \
    harfbuzz \
    ca-certificates \
    ttf-freefont \
    && rm -rf /var/cache/apk/*

# Tell Puppeteer to skip installing Chrome
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source code
COPY --chown=nextjs:nodejs . .

# Create test directories with proper permissions
RUN mkdir -p tests/e2e/screenshots tests/e2e/videos tests/reports \
    && chown -R nextjs:nodejs tests/

# Expose test port
EXPOSE 3002

# Health check for testing
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3002/api/health || exit 1

# Switch to non-root user
USER nextjs

# Default command for test environment
CMD ["yarn", "dev", "--hostname", "0.0.0.0", "--port", "3002"]

# Staging stage (production-like with additional debugging tools)
FROM production AS staging

# Switch back to root to install debugging tools
USER root

# Install debugging and monitoring tools
RUN apk add --no-cache \
    htop \
    strace \
    tcpdump \
    && rm -rf /var/cache/apk/*

# Set staging environment
ENV NODE_ENV=staging
ENV NEXT_TELEMETRY_DISABLED=1

# Switch back to nextjs user
USER nextjs

# Use same command as production
CMD ["node", "server.js"]