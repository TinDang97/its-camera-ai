'use client'

import { TrafficMetricsCard } from '@/components/features/dashboard/TrafficMetricsCard'
import { AlertPanel } from '@/components/features/dashboard/AlertPanel'
import { CameraGridView } from '@/components/features/camera/CameraGridView'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { IconActivity, IconCamera, IconAlertTriangle, IconTrendingUp } from '@tabler/icons-react'

export default function DashboardPage() {
  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Traffic Monitoring Dashboard</h1>
          <p className="text-muted-foreground">Real-time traffic analytics and camera monitoring</p>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-muted-foreground">Last updated:</span>
          <span className="text-sm font-medium">{new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Vehicles</CardTitle>
            <IconActivity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12,543</div>
            <p className="text-xs text-muted-foreground">+20.1% from last hour</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Cameras</CardTitle>
            <IconCamera className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">48/52</div>
            <p className="text-xs text-muted-foreground">92.3% online</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <IconAlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">7</div>
            <p className="text-xs text-muted-foreground">3 critical, 4 warnings</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Speed</CardTitle>
            <IconTrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">45 km/h</div>
            <p className="text-xs text-muted-foreground">-5 km/h from average</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="cameras">Live Cameras</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <TrafficMetricsCard />
            <AlertPanel maxHeight="320px" />
          </div>
          <CameraGridView selectedLayout="2x2" />
        </TabsContent>

        <TabsContent value="cameras">
          <CameraGridView selectedLayout="3x3" />
        </TabsContent>

        <TabsContent value="alerts">
          <AlertPanel maxHeight="600px" />
        </TabsContent>

        <TabsContent value="analytics">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Traffic Flow Heatmap</CardTitle>
                <CardDescription>Vehicle density across monitored zones</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
                <span className="text-muted-foreground">Heatmap Visualization</span>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Hourly Traffic Trends</CardTitle>
                <CardDescription>Vehicle count over the last 24 hours</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
                <span className="text-muted-foreground">Time Series Chart</span>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
