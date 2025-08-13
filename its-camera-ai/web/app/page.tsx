import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Activity, Camera, Car, AlertTriangle, TrendingUp, Users, Clock, Shield } from 'lucide-react'

export default function Dashboard() {
  const stats = [
    {
      title: 'Active Cameras',
      value: '12',
      change: '+2',
      icon: Camera,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100'
    },
    {
      title: 'Vehicles Detected',
      value: '2,847',
      change: '+15%',
      icon: Car,
      color: 'text-green-600',
      bgColor: 'bg-green-100'
    },
    {
      title: 'Active Alerts',
      value: '3',
      change: '-25%',
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-100'
    },
    {
      title: 'Avg Speed',
      value: '35 mph',
      change: '-2 mph',
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100'
    }
  ]

  const cameras = [
    { id: 1, name: 'Main Street North', status: 'online', vehicles: 127 },
    { id: 2, name: 'Highway 101 West', status: 'online', vehicles: 342 },
    { id: 3, name: 'Downtown Bridge', status: 'maintenance', vehicles: 0 },
    { id: 4, name: 'School Zone', status: 'offline', vehicles: 0 }
  ]

  const alerts = [
    { id: 1, type: 'speed', message: 'Speed violation on Main Street', time: '2 min ago', severity: 'high' },
    { id: 2, type: 'congestion', message: 'Heavy traffic on Highway 101', time: '5 min ago', severity: 'medium' },
    { id: 3, type: 'system', message: 'Camera 3 requires maintenance', time: '1 hour ago', severity: 'low' }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Camera className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-bold">ITS Camera AI</h1>
            </div>
            <nav className="flex space-x-4">
              <Button variant="ghost">Dashboard</Button>
              <Button variant="ghost">Cameras</Button>
              <Button variant="ghost">Analytics</Button>
              <Button variant="ghost">Settings</Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Title */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Traffic Monitoring Dashboard</h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Real-time traffic analytics and vehicle tracking</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat) => {
            const Icon = stat.icon
            return (
              <Card key={stat.title}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
                  <div className={`p-2 rounded-full ${stat.bgColor}`}>
                    <Icon className={`h-4 w-4 ${stat.color}`} />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <p className="text-xs text-muted-foreground mt-1">{stat.change} from last hour</p>
                </CardContent>
              </Card>
            )
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Status */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Camera Status</CardTitle>
                <CardDescription>Live monitoring of all camera feeds</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {cameras.map((camera) => (
                    <div key={camera.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                          camera.status === 'online' ? 'bg-green-500' :
                          camera.status === 'maintenance' ? 'bg-yellow-500' : 'bg-red-500'
                        }`} />
                        <div>
                          <p className="font-medium">{camera.name}</p>
                          <p className="text-sm text-gray-500">{camera.status}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">{camera.vehicles}</p>
                        <p className="text-xs text-gray-500">vehicles/hr</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Alerts */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Recent Alerts</CardTitle>
                <CardDescription>System notifications and warnings</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {alerts.map((alert) => (
                    <div key={alert.id} className="flex items-start space-x-3">
                      <div className={`mt-0.5 w-2 h-2 rounded-full ${
                        alert.severity === 'high' ? 'bg-red-500' :
                        alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                      }`} />
                      <div className="flex-1">
                        <p className="text-sm font-medium">{alert.message}</p>
                        <p className="text-xs text-gray-500">{alert.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Traffic Chart Placeholder */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Traffic Flow Analysis</CardTitle>
            <CardDescription>Hourly vehicle count across all cameras</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
              <p className="text-gray-500">Traffic chart will be implemented with Recharts</p>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
