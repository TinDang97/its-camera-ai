'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Maximize2, Grid, Grid3x3, Square, Activity, WifiOff, Wifi } from 'lucide-react'

export interface Camera {
  id: string
  name: string
  location: string
  status: 'online' | 'offline' | 'maintenance'
  health: 'good' | 'degraded' | 'poor'
  fps: number
  resolution: string
  streamUrl: string
  thumbnail?: string
  vehicleCount?: number
  lastDetection?: Date
}

interface CameraGridViewProps {
  cameras?: Camera[]
  onCameraSelect?: (camera: Camera) => void
  selectedLayout?: '1x1' | '2x2' | '3x3' | '4x4'
}

// Mock camera data
const mockCameras: Camera[] = [
  {
    id: 'CAM-001',
    name: 'Main Intersection North',
    location: 'Main St & 5th Ave',
    status: 'online',
    health: 'good',
    fps: 30,
    resolution: '1920x1080',
    streamUrl: '/api/stream/cam-001',
    vehicleCount: 42,
    lastDetection: new Date()
  },
  {
    id: 'CAM-002',
    name: 'Highway Entry Ramp',
    location: 'I-95 North Entry',
    status: 'online',
    health: 'good',
    fps: 25,
    resolution: '1920x1080',
    streamUrl: '/api/stream/cam-002',
    vehicleCount: 128,
    lastDetection: new Date()
  },
  {
    id: 'CAM-003',
    name: 'Downtown Plaza',
    location: 'City Center',
    status: 'online',
    health: 'degraded',
    fps: 20,
    resolution: '1280x720',
    streamUrl: '/api/stream/cam-003',
    vehicleCount: 15,
    lastDetection: new Date(Date.now() - 60000)
  },
  {
    id: 'CAM-004',
    name: 'School Zone Monitor',
    location: 'Oak Elementary',
    status: 'offline',
    health: 'poor',
    fps: 0,
    resolution: '1920x1080',
    streamUrl: '/api/stream/cam-004',
    vehicleCount: 0
  },
  {
    id: 'CAM-005',
    name: 'Parking Garage A',
    location: 'Level 2',
    status: 'online',
    health: 'good',
    fps: 15,
    resolution: '1280x720',
    streamUrl: '/api/stream/cam-005',
    vehicleCount: 89
  },
  {
    id: 'CAM-006',
    name: 'Transit Hub',
    location: 'Bus Terminal',
    status: 'online',
    health: 'good',
    fps: 30,
    resolution: '1920x1080',
    streamUrl: '/api/stream/cam-006',
    vehicleCount: 23
  }
]

export function CameraGridView({
  cameras = mockCameras,
  onCameraSelect,
  selectedLayout = '2x2'
}: CameraGridViewProps) {
  const [layout, setLayout] = useState(selectedLayout)
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null)

  const getGridClass = () => {
    switch (layout) {
      case '1x1': return 'grid-cols-1'
      case '2x2': return 'grid-cols-2'
      case '3x3': return 'grid-cols-3'
      case '4x4': return 'grid-cols-4'
      default: return 'grid-cols-2'
    }
  }

  const getVisibleCameras = () => {
    const layoutMap = {
      '1x1': 1,
      '2x2': 4,
      '3x3': 9,
      '4x4': 16
    }
    return cameras.slice(0, layoutMap[layout])
  }

  const getStatusColor = (status: Camera['status']) => {
    switch (status) {
      case 'online': return 'default'
      case 'offline': return 'destructive'
      case 'maintenance': return 'secondary'
    }
  }

  const getHealthColor = (health: Camera['health']) => {
    switch (health) {
      case 'good': return 'text-green-600'
      case 'degraded': return 'text-yellow-600'
      case 'poor': return 'text-red-600'
    }
  }

  const handleCameraClick = (camera: Camera) => {
    setSelectedCamera(camera.id)
    onCameraSelect?.(camera)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Camera Feeds</CardTitle>
          <div className="flex items-center space-x-2">
            <Select value={layout} onValueChange={(value: any) => setLayout(value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1x1">
                  <div className="flex items-center space-x-2">
                    <Square className="h-4 w-4" />
                    <span>1×1</span>
                  </div>
                </SelectItem>
                <SelectItem value="2x2">
                  <div className="flex items-center space-x-2">
                    <Grid className="h-4 w-4" />
                    <span>2×2</span>
                  </div>
                </SelectItem>
                <SelectItem value="3x3">
                  <div className="flex items-center space-x-2">
                    <Grid3x3 className="h-4 w-4" />
                    <span>3×3</span>
                  </div>
                </SelectItem>
                <SelectItem value="4x4">
                  <div className="flex items-center space-x-2">
                    <Grid3x3 className="h-4 w-4" />
                    <span>4×4</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className={`grid ${getGridClass()} gap-4`}>
          {getVisibleCameras().map((camera) => (
            <div
              key={camera.id}
              className={`relative border rounded-lg overflow-hidden cursor-pointer transition-all hover:ring-2 hover:ring-primary ${
                selectedCamera === camera.id ? 'ring-2 ring-primary' : ''
              }`}
              onClick={() => handleCameraClick(camera)}
            >
              {/* Camera Feed Placeholder */}
              <div className="aspect-video bg-gray-900 relative">
                {camera.status === 'online' ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-white text-opacity-20 text-6xl font-bold">
                      {camera.id}
                    </div>
                    {/* Simulated video overlay */}
                    <div className="absolute inset-0 bg-gradient-to-br from-transparent to-black opacity-30" />
                  </div>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                    <WifiOff className="h-12 w-12 text-gray-500" />
                  </div>
                )}
                
                {/* Camera Info Overlay */}
                <div className="absolute top-2 left-2 right-2 flex justify-between">
                  <div className="flex items-center space-x-2">
                    <Badge variant={getStatusColor(camera.status)}>
                      {camera.status === 'online' ? (
                        <Wifi className="h-3 w-3 mr-1" />
                      ) : (
                        <WifiOff className="h-3 w-3 mr-1" />
                      )}
                      {camera.status}
                    </Badge>
                    {camera.status === 'online' && (
                      <Badge variant="outline" className="bg-black/50 text-white border-white/20">
                        {camera.fps} FPS
                      </Badge>
                    )}
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0 bg-black/50 hover:bg-black/70 text-white"
                    onClick={(e) => {
                      e.stopPropagation()
                      // Handle fullscreen
                    }}
                  >
                    <Maximize2 className="h-3 w-3" />
                  </Button>
                </div>
                
                {/* Bottom Info Bar */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                  <div className="text-white">
                    <div className="font-semibold text-sm">{camera.name}</div>
                    <div className="text-xs opacity-75">{camera.location}</div>
                    {camera.status === 'online' && (
                      <div className="flex items-center justify-between mt-1">
                        <div className="flex items-center space-x-2 text-xs">
                          <Activity className={`h-3 w-3 ${getHealthColor(camera.health)}`} />
                          <span className={getHealthColor(camera.health)}>
                            {camera.health}
                          </span>
                        </div>
                        {camera.vehicleCount !== undefined && (
                          <div className="text-xs">
                            {camera.vehicleCount} vehicles
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}