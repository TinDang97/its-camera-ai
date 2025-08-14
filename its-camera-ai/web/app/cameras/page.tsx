'use client'

import { useState } from 'react'
import { CameraGridView, Camera } from '@/components/features/camera/CameraGridView'
import { LiveStreamPlayer } from '@/components/features/camera/LiveStreamPlayer'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Search, Plus, Settings, Filter, Camera as CameraIcon,
  Wifi, WifiOff, Activity, MapPin, Clock, Maximize2
} from 'lucide-react'

export default function CamerasPage() {
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [locationFilter, setLocationFilter] = useState('all')

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
      name: 'Bridge Overpass',
      location: 'River Bridge',
      status: 'maintenance',
      health: 'good',
      fps: 0,
      resolution: '1920x1080',
      streamUrl: '/api/stream/cam-005',
      vehicleCount: 0
    }
  ]

  const filteredCameras = mockCameras.filter(camera => {
    const matchesSearch = camera.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         camera.location.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || camera.status === statusFilter
    return matchesSearch && matchesStatus
  })

  const getStatusIcon = (status: Camera['status']) => {
    switch (status) {
      case 'online': return <Wifi className="h-4 w-4 text-green-600" />
      case 'offline': return <WifiOff className="h-4 w-4 text-red-600" />
      case 'maintenance': return <Settings className="h-4 w-4 text-orange-600" />
    }
  }

  const getHealthColor = (health: Camera['health']) => {
    switch (health) {
      case 'good': return 'bg-green-100 text-green-800'
      case 'degraded': return 'bg-yellow-100 text-yellow-800'
      case 'poor': return 'bg-red-100 text-red-800'
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Camera Management</h1>
          <p className="text-muted-foreground">Monitor and manage all traffic cameras</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Add Camera
          </Button>
          <Button variant="outline">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search cameras by name or location..."
                  className="pl-8"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="online">Online</SelectItem>
                <SelectItem value="offline">Offline</SelectItem>
                <SelectItem value="maintenance">Maintenance</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <Tabs defaultValue="grid" className="space-y-4">
        <TabsList>
          <TabsTrigger value="grid">Grid View</TabsTrigger>
          <TabsTrigger value="list">List View</TabsTrigger>
          <TabsTrigger value="single">Single View</TabsTrigger>
        </TabsList>

        <TabsContent value="grid">
          <CameraGridView 
            cameras={filteredCameras}
            onCameraSelect={setSelectedCamera}
          />
        </TabsContent>

        <TabsContent value="list">
          <Card>
            <CardHeader>
              <CardTitle>Camera List</CardTitle>
              <CardDescription>Detailed view of all cameras</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredCameras.map((camera) => (
                  <div
                    key={camera.id}
                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 cursor-pointer"
                    onClick={() => setSelectedCamera(camera)}
                  >
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(camera.status)}
                        <CameraIcon className="h-6 w-6 text-muted-foreground" />
                      </div>
                      <div>
                        <h3 className="font-semibold">{camera.name}</h3>
                        <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                          <MapPin className="h-3 w-3" />
                          <span>{camera.location}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <div className="text-sm font-medium">{camera.resolution}</div>
                        <div className="text-xs text-muted-foreground">{camera.fps} FPS</div>
                      </div>
                      <Badge className={getHealthColor(camera.health)}>
                        {camera.health}
                      </Badge>
                      <Badge variant="outline">
                        {camera.vehicleCount} vehicles
                      </Badge>
                      <Button size="sm" variant="ghost">
                        <Maximize2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="single">
          <div className="space-y-4">
            {selectedCamera ? (
              <LiveStreamPlayer
                streamUrl={selectedCamera.streamUrl}
                cameraId={selectedCamera.id}
                cameraName={selectedCamera.name}
              />
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <CameraIcon className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Camera Selected</h3>
                  <p className="text-muted-foreground">Select a camera from the grid or list to view its stream</p>
                </CardContent>
              </Card>
            )}
            
            {/* Camera Details */}
            {selectedCamera && (
              <Card>
                <CardHeader>
                  <CardTitle>Camera Details: {selectedCamera.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Status</label>
                      <div className="flex items-center space-x-2 mt-1">
                        {getStatusIcon(selectedCamera.status)}
                        <span className="capitalize">{selectedCamera.status}</span>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Health</label>
                      <div className="mt-1">
                        <Badge className={getHealthColor(selectedCamera.health)}>
                          {selectedCamera.health}
                        </Badge>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Resolution</label>
                      <div className="mt-1 font-medium">{selectedCamera.resolution}</div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Frame Rate</label>
                      <div className="mt-1 font-medium">{selectedCamera.fps} FPS</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}