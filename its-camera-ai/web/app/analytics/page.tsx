'use client'

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

export default function AnalyticsPage() {
  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Traffic Analytics</h1>
          <p className="text-muted-foreground">Advanced traffic analytics and reporting</p>
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

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Traffic Volume</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">142,543</div>
            <p className="text-xs text-muted-foreground">+12.5% from last week</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Peak Hour Traffic</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8:30 AM</div>
            <p className="text-xs text-muted-foreground">2,543 vehicles/hour</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Speed</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">42 km/h</div>
            <p className="text-xs text-muted-foreground">-3 km/h from average</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Incidents</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">-25% from yesterday</p>
          </CardContent>
        </Card>
      </div>

      {/* Analytics Tabs */}
      <Tabs defaultValue="traffic-flow" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="traffic-flow">Traffic Flow</TabsTrigger>
          <TabsTrigger value="vehicle-types">Vehicle Types</TabsTrigger>
          <TabsTrigger value="speed-analysis">Speed Analysis</TabsTrigger>
          <TabsTrigger value="incidents">Incidents</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
        </TabsList>

        <TabsContent value="traffic-flow" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Traffic Volume Trends</CardTitle>
                <CardDescription>Hourly traffic volume over the last 24 hours</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <span className="text-muted-foreground">Traffic Volume Chart</span>
                  <p className="text-sm text-muted-foreground mt-2">Real-time data visualization will be implemented</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Traffic Flow Heatmap</CardTitle>
                <CardDescription>Geographic traffic density across monitored zones</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
                <div className="text-center">
                  <MapPin className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <span className="text-muted-foreground">Traffic Heatmap</span>
                  <p className="text-sm text-muted-foreground mt-2">Geographic visualization will be implemented</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="vehicle-types" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Vehicle Classification</CardTitle>
              <CardDescription>Distribution of vehicle types detected</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
              <div className="text-center">
                <Activity className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <span className="text-muted-foreground">Vehicle Classification Chart</span>
                <p className="text-sm text-muted-foreground mt-2">AI model classification results</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="speed-analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Speed Distribution</CardTitle>
              <CardDescription>Speed analysis across different zones and time periods</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
              <div className="text-center">
                <TrendingUp className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <span className="text-muted-foreground">Speed Analysis Chart</span>
                <p className="text-sm text-muted-foreground mt-2">Speed trend analysis and statistics</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="incidents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Incident Analysis</CardTitle>
              <CardDescription>Traffic incidents and their impact on flow</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
              <div className="text-center">
                <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <span className="text-muted-foreground">Incident Analysis</span>
                <p className="text-sm text-muted-foreground mt-2">Incident patterns and response metrics</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Traffic Predictions</CardTitle>
              <CardDescription>AI-powered traffic flow predictions and forecasting</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px] flex items-center justify-center bg-muted/10 rounded">
              <div className="text-center">
                <Activity className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <span className="text-muted-foreground">Predictive Analytics</span>
                <p className="text-sm text-muted-foreground mt-2">AI model predictions with confidence intervals</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}