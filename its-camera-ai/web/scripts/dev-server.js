#!/usr/bin/env node

/**
 * Development server script that provides mock API endpoints
 * This allows the web app to run independently during development
 */

const express = require('express')
const { createServer } = require('http')
const { WebSocketServer } = require('ws')
const cors = require('cors')

const app = express()
const server = createServer(app)
const wss = new WebSocketServer({ server })

// Middleware
app.use(cors())
app.use(express.json())

// Mock data generators
function generateMockTrafficData() {
  return {
    totalVehicles: Math.floor(Math.random() * 500) + 1000,
    avgSpeed: Math.floor(Math.random() * 30) + 30,
    congestionLevel: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
    timestamp: new Date().toISOString()
  }
}

function generateMockAlert() {
  const severities = ['low', 'medium', 'high', 'critical']
  const types = ['congestion', 'incident', 'speed_violation', 'camera_offline']
  const locations = ['Main St & 5th Ave', 'I-95 North Entry', 'Downtown Plaza', 'School Zone', 'Bridge Overpass']
  
  return {
    id: Date.now().toString(),
    severity: severities[Math.floor(Math.random() * severities.length)],
    type: types[Math.floor(Math.random() * types.length)],
    message: 'Traffic event detected',
    location: locations[Math.floor(Math.random() * locations.length)],
    timestamp: new Date().toISOString()
  }
}

function generateMockCameraData() {
  return {
    cameraId: `CAM-${Math.floor(Math.random() * 100).toString().padStart(3, '0')}`,
    status: Math.random() > 0.9 ? 'offline' : 'online',
    fps: Math.floor(Math.random() * 5) + 25,
    detections: Math.floor(Math.random() * 50),
    health: Math.random() > 0.8 ? 'degraded' : 'good',
    timestamp: new Date().toISOString()
  }
}

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

app.get('/api/traffic/metrics', (req, res) => {
  res.json(generateMockTrafficData())
})

app.get('/api/alerts', (req, res) => {
  const alerts = Array.from({ length: 5 }, () => generateMockAlert())
  res.json({ alerts })
})

app.get('/api/cameras', (req, res) => {
  const cameras = Array.from({ length: 10 }, (_, i) => ({
    id: `CAM-${(i + 1).toString().padStart(3, '0')}`,
    name: `Camera ${i + 1}`,
    location: `Location ${i + 1}`,
    status: Math.random() > 0.9 ? 'offline' : 'online',
    health: Math.random() > 0.8 ? 'degraded' : 'good',
    fps: Math.floor(Math.random() * 5) + 25,
    resolution: '1920x1080',
    streamUrl: `/api/stream/cam-${(i + 1).toString().padStart(3, '0')}`,
    vehicleCount: Math.floor(Math.random() * 100),
    lastDetection: new Date()
  }))
  res.json({ cameras })
})

// WebSocket handling
wss.on('connection', (ws) => {
  console.log('WebSocket client connected')
  
  // Send periodic updates
  const interval = setInterval(() => {
    const messageTypes = ['traffic', 'alert', 'camera', 'model']
    const type = messageTypes[Math.floor(Math.random() * messageTypes.length)]
    
    let data
    switch (type) {
      case 'traffic':
        data = generateMockTrafficData()
        break
      case 'alert':
        data = generateMockAlert()
        break
      case 'camera':
        data = generateMockCameraData()
        break
      case 'model':
        data = {
          modelId: 'yolo11-traffic-v2',
          accuracy: (Math.random() * 5 + 93).toFixed(2),
          latency: Math.floor(Math.random() * 30) + 20,
          throughput: Math.floor(Math.random() * 50) + 100,
          timestamp: new Date().toISOString()
        }
        break
    }
    
    ws.send(JSON.stringify({
      type,
      data,
      timestamp: new Date().toISOString()
    }))
  }, 2000)
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected')
    clearInterval(interval)
  })
  
  ws.on('error', (error) => {
    console.error('WebSocket error:', error)
    clearInterval(interval)
  })
})

const PORT = process.env.PORT || 8000

server.listen(PORT, () => {
  console.log(`🚀 Development server running on http://localhost:${PORT}`)
  console.log(`📡 WebSocket server running on ws://localhost:${PORT}/ws`)
  console.log('📊 Mock data endpoints available:')
  console.log('  - GET /api/health')
  console.log('  - GET /api/traffic/metrics')
  console.log('  - GET /api/alerts')
  console.log('  - GET /api/cameras')
})

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Shutting down development server...')
  server.close(() => {
    console.log('✅ Server closed')
    process.exit(0)
  })
})