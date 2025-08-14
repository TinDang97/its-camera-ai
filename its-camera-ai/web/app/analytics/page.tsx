'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { 
  TrendingUp, 
  BarChart3, 
  Activity, 
  Clock, 
  MapPin, 
  AlertTriangle,
  Download,
  Calendar,
  Filter
} from 'lucide-react'
import { RealTimeMetrics } from '@/components/analytics/real-time-metrics'
import { TrafficFlowChart } from '@/components/analytics/traffic-flow-chart'
import { TrafficHeatmap } from '@/components/analytics/traffic-heatmap'
import { IncidentManagement } from '@/components/analytics/incident-management'
import { useAnalyticsWebSocket } from '@/hooks/use-analytics-websocket'

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'1h' | '4h' | '24h' | '7d'>('24h')
  const [chartType, setChartType] = useState<'area' | 'line' | 'bar' | 'composed'>('area')
  const { metrics, incidents, vehicleCounts, speedUpdates, predictions, isConnected } = useAnalyticsWebSocket()

  // Generate mock data for charts (in production, this would come from API)
  const generateMockTrafficData = () => {
    const data = []
    const now = new Date()
    const points = timeRange === '1h' ? 12 : timeRange === '4h' ? 48 : timeRange === '24h' ? 24 : 168
    
    for (let i = points - 1; i >= 0; i--) {
      const timestamp = new Date(now)
      if (timeRange === '1h') {
        timestamp.setMinutes(timestamp.getMinutes() - i * 5)
      } else if (timeRange === '4h') {
        timestamp.setMinutes(timestamp.getMinutes() - i * 5)
      } else if (timeRange === '24h') {
        timestamp.setHours(timestamp.getHours() - i)
      } else {
        timestamp.setHours(timestamp.getHours() - i)
      }
      
      const hour = timestamp.getHours()
      const isRushHour = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)
      const baseCount = isRushHour ? 45 : hour >= 22 || hour <= 6 ? 10 : 25
      
      data.push({
        timestamp: timestamp.toISOString(),
        vehicleCount: Math.floor(baseCount + Math.random() * 15 - 7),
        averageSpeed: isRushHour ? 35 + Math.random() * 10 : 50 + Math.random() * 15,
        congestionLevel: isRushHour ? (Math.random() > 0.3 ? 'high' : 'moderate') : 'low',
        occupancy: Math.min(100, baseCount * 2 + Math.random() * 20)
      })
    }
    
    return data
  }

  const generateMockHeatmapData = () => {
    const cameras = [
      { id: 'CAM001', name: 'Main St & 1st Ave', lat: 40.7128, lng: -74.0060, intersection: 'Main St & 1st Ave' },
      { id: 'CAM002', name: '2nd Ave & Oak Dr', lat: 40.7200, lng: -74.0100, intersection: '2nd Ave & Oak Dr' },
      { id: 'CAM003', name: 'Broadway & 5th St', lat: 40.7150, lng: -74.0050, intersection: 'Broadway & 5th St' },
      { id: 'CAM004', name: 'Park Ave & Elm St', lat: 40.7180, lng: -74.0080, intersection: 'Park Ave & Elm St' },
      { id: 'CAM005', name: 'Highway 101 North', lat: 40.7250, lng: -74.0120, intersection: 'Highway 101 North' }
    ]
    
    const data = []
    const now = new Date()
    const hours = timeRange === '1h' ? 4 : timeRange === '4h' ? 8 : 24
    
    cameras.forEach(camera => {
      for (let i = 0; i < hours; i++) {
        const timestamp = new Date(now)
        timestamp.setHours(timestamp.getHours() - i)
        
        const hour = timestamp.getHours()
        const isRushHour = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)
        
        data.push({
          cameraId: camera.id,
          cameraName: camera.name,
          location: {
            lat: camera.lat,
            lng: camera.lng,
            intersection: camera.intersection
          },
          timestamp: timestamp.toISOString(),
          vehicleCount: Math.floor(Math.random() * 50) + (isRushHour ? 30 : 10),
          averageSpeed: isRushHour ? 30 + Math.random() * 15 : 45 + Math.random() * 20,
          congestionLevel: isRushHour ? (Math.random() > 0.4 ? 'high' : 'moderate') : 'low',
          incidentCount: Math.floor(Math.random() * 3)
        })
      }
    })
    
    return data
  }

  const generateMockIncidents = () => {
    const types = ['accident', 'congestion', 'road_closure', 'weather', 'construction']
    const severities = ['low', 'medium', 'high', 'critical']
    const statuses = ['active', 'monitoring', 'resolved']
    
    return Array.from({ length: 8 }, (_, i) => ({
      id: `INC00${i + 1}`,
      type: types[Math.floor(Math.random() * types.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
      status: statuses[Math.floor(Math.random() * statuses.length)],
      cameraId: `CAM00${Math.floor(Math.random() * 5) + 1}`,
      cameraName: `Camera ${Math.floor(Math.random() * 5) + 1}`,
      location: {
        lat: 40.7128 + Math.random() * 0.01,
        lng: -74.0060 + Math.random() * 0.01,
        intersection: `Street ${i + 1} & Avenue ${String.fromCharCode(65 + i)}`
      },
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      description: `Traffic incident on Street ${i + 1}`,
      affectedLanes: Math.floor(Math.random() * 3) + 1,
      vehiclesAffected: Math.floor(Math.random() * 20) + 5,
      responders: {
        police: Math.random() > 0.5,
        ambulance: Math.random() > 0.7,
        fire: Math.random() > 0.8,
        towing: Math.random() > 0.6
      }
    }))
  }

  const [trafficData, setTrafficData] = useState(generateMockTrafficData())
  const [heatmapData, setHeatmapData] = useState(generateMockHeatmapData())
  const [incidentData, setIncidentData] = useState(generateMockIncidents())

  // Update data when time range changes
  useEffect(() => {
    setTrafficData(generateMockTrafficData())
    setHeatmapData(generateMockHeatmapData())
  }, [timeRange])

  const handleTimeRangeChange = (newRange: '1h' | '4h' | '24h' | '7d') => {
    setTimeRange(newRange)
  }

  const handleIncidentResolve = (incidentId: string) => {
    setIncidentData(prev => 
      prev.map(inc => 
        inc.id === incidentId ? { ...inc, status: 'resolved' } : inc
      )
    )
  }
  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Traffic Analytics</h1>
          <p className="text-muted-foreground">
            Advanced traffic analytics and reporting
            {isConnected && (
              <span className="ml-2 text-green-600 text-sm">● Connected</span>
            )}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline">
            <Calendar className="h-4 w-4 mr-2" />
            Date Range
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Real-Time Metrics Component */}
      <RealTimeMetrics />

      {/* Analytics Tabs */}
      <Tabs defaultValue="traffic-flow" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="traffic-flow">Traffic Flow</TabsTrigger>
          <TabsTrigger value="heatmap">Density Heatmap</TabsTrigger>
          <TabsTrigger value="incidents">Incidents</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
        </TabsList>

        <TabsContent value="traffic-flow" className="space-y-4">
          <TrafficFlowChart
            data={trafficData}
            timeRange={timeRange}
            chartType={chartType}
            showPredictions={false}
            onTimeRangeChange={handleTimeRangeChange}
          />
          
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Chart Type</CardTitle>
                <CardDescription>Select visualization type</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2">
                  <Button
                    variant={chartType === 'area' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setChartType('area')}
                  >
                    Area
                  </Button>
                  <Button
                    variant={chartType === 'line' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setChartType('line')}
                  >
                    Line
                  </Button>
                  <Button
                    variant={chartType === 'bar' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setChartType('bar')}
                  >
                    Bar
                  </Button>
                  <Button
                    variant={chartType === 'composed' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setChartType('composed')}
                  >
                    Combined
                  </Button>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Data Summary</CardTitle>
                <CardDescription>Current period statistics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total Vehicles</span>
                    <span className="font-medium">
                      {trafficData.reduce((sum, d) => sum + d.vehicleCount, 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Avg Speed</span>
                    <span className="font-medium">
                      {(trafficData.reduce((sum, d) => sum + d.averageSpeed, 0) / trafficData.length).toFixed(1)} km/h
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Peak Congestion</span>
                    <span className="font-medium">
                      {trafficData.filter(d => d.congestionLevel === 'high').length} hours
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="heatmap" className="space-y-4">
          <TrafficHeatmap
            data={heatmapData}
            timeRange={timeRange}
            onTimeRangeChange={handleTimeRangeChange}
          />
        </TabsContent>

        <TabsContent value="incidents" className="space-y-4">
          <IncidentManagement
            incidents={incidentData}
            onIncidentResolve={handleIncidentResolve}
          />
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <TrafficFlowChart
            data={trafficData}
            timeRange={timeRange}
            chartType="area"
            showPredictions={true}
            predictions={[
              ...trafficData.slice(-4).map(d => ({
                ...d,
                timestamp: new Date(new Date(d.timestamp).getTime() + 3600000).toISOString(),
                vehicleCount: Math.floor(d.vehicleCount * 1.1),
                averageSpeed: d.averageSpeed * 0.95
              }))
            ]}
            onTimeRangeChange={handleTimeRangeChange}
          />
          
          <Card>
            <CardHeader>
              <CardTitle>Prediction Accuracy</CardTitle>
              <CardDescription>Model performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Accuracy</p>
                  <p className="text-2xl font-bold">92.3%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">MAE</p>
                  <p className="text-2xl font-bold">3.4</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">RMSE</p>
                  <p className="text-2xl font-bold">5.2</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">R² Score</p>
                  <p className="text-2xl font-bold">0.89</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Generate Reports</CardTitle>
                <CardDescription>Create custom traffic analysis reports</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Button className="w-full" variant="outline">
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Daily Traffic Summary
                  </Button>
                  <Button className="w-full" variant="outline">
                    <TrendingUp className="h-4 w-4 mr-2" />
                    Weekly Performance Report
                  </Button>
                  <Button className="w-full" variant="outline">
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    Incident Analysis Report
                  </Button>
                  <Button className="w-full" variant="outline">
                    <Activity className="h-4 w-4 mr-2" />
                    Monthly Analytics Overview
                  </Button>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Recent Reports</CardTitle>
                <CardDescription>Previously generated reports</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 rounded hover:bg-muted/50">
                    <div>
                      <p className="text-sm font-medium">Weekly Report - Week 45</p>
                      <p className="text-xs text-muted-foreground">Generated 2 days ago</p>
                    </div>
                    <Button size="sm" variant="ghost">
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded hover:bg-muted/50">
                    <div>
                      <p className="text-sm font-medium">October Performance</p>
                      <p className="text-xs text-muted-foreground">Generated 1 week ago</p>
                    </div>
                    <Button size="sm" variant="ghost">
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded hover:bg-muted/50">
                    <div>
                      <p className="text-sm font-medium">Q3 2024 Summary</p>
                      <p className="text-xs text-muted-foreground">Generated 2 weeks ago</p>
                    </div>
                    <Button size="sm" variant="ghost">
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}