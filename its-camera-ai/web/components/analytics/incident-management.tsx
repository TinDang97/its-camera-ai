'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock, 
  MapPin, 
  Camera,
  TrendingUp,
  Search,
  Filter,
  Download,
  RefreshCw,
  ChevronRight,
  AlertCircle,
  Activity
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { format, formatDistanceToNow } from 'date-fns'
import { cn } from '@/lib/utils'
import { useAnalyticsWebSocket } from '@/hooks/use-analytics-websocket'

interface Incident {
  id: string
  type: 'accident' | 'congestion' | 'road_closure' | 'weather' | 'construction' | 'other'
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'active' | 'monitoring' | 'resolved' | 'closed'
  cameraId: string
  cameraName: string
  location: {
    lat: number
    lng: number
    intersection?: string
    direction?: string
  }
  timestamp: string
  resolvedAt?: string
  description: string
  affectedLanes?: number
  estimatedClearTime?: string
  vehiclesAffected?: number
  responders?: {
    police?: boolean
    ambulance?: boolean
    fire?: boolean
    towing?: boolean
  }
  updates?: {
    timestamp: string
    message: string
    updatedBy?: string
  }[]
}

interface IncidentManagementProps {
  incidents?: Incident[]
  onIncidentSelect?: (incident: Incident) => void
  onIncidentResolve?: (incidentId: string) => void
  onIncidentUpdate?: (incidentId: string, update: any) => void
  loading?: boolean
}

export function IncidentManagement({
  incidents: propIncidents,
  onIncidentSelect,
  onIncidentResolve,
  onIncidentUpdate,
  loading = false
}: IncidentManagementProps) {
  const { incidents: wsIncidents, isConnected } = useAnalyticsWebSocket()
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null)
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'monitoring' | 'resolved'>('active')
  const [filterSeverity, setFilterSeverity] = useState<'all' | 'low' | 'medium' | 'high' | 'critical'>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'time' | 'severity'>('time')

  // Combine prop incidents with WebSocket incidents
  const allIncidents = propIncidents || []
  
  // Convert WebSocket incidents to full incident format
  const formattedWsIncidents: Incident[] = wsIncidents.map(inc => ({
    id: inc.id,
    type: inc.type as Incident['type'],
    severity: inc.severity as Incident['severity'],
    status: 'active' as const,
    cameraId: inc.cameraId,
    cameraName: `Camera ${inc.cameraId}`,
    location: inc.location,
    timestamp: inc.timestamp,
    description: inc.description,
    affectedLanes: inc.affectedLanes,
    vehiclesAffected: inc.vehiclesAffected
  }))

  const combinedIncidents = [...allIncidents, ...formattedWsIncidents]
    .filter((incident, index, self) => 
      index === self.findIndex(i => i.id === incident.id)
    )

  // Filter and sort incidents
  const filteredIncidents = combinedIncidents
    .filter(incident => {
      if (filterStatus !== 'all' && incident.status !== filterStatus) return false
      if (filterSeverity !== 'all' && incident.severity !== filterSeverity) return false
      if (searchQuery && !incident.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !incident.cameraName.toLowerCase().includes(searchQuery.toLowerCase())) return false
      return true
    })
    .sort((a, b) => {
      if (sortBy === 'severity') {
        const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 }
        return severityOrder[a.severity] - severityOrder[b.severity]
      }
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    })

  // Calculate statistics
  const stats = {
    total: combinedIncidents.length,
    active: combinedIncidents.filter(i => i.status === 'active').length,
    critical: combinedIncidents.filter(i => i.severity === 'critical').length,
    resolved: combinedIncidents.filter(i => i.status === 'resolved').length,
    avgResolutionTime: '45 min' // Mock value
  }

  const getSeverityColor = (severity: Incident['severity']) => {
    switch (severity) {
      case 'critical': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-blue-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusIcon = (status: Incident['status']) => {
    switch (status) {
      case 'active': return <AlertCircle className="w-4 h-4" />
      case 'monitoring': return <Activity className="w-4 h-4" />
      case 'resolved': return <CheckCircle className="w-4 h-4" />
      case 'closed': return <XCircle className="w-4 h-4" />
      default: return null
    }
  }

  const getIncidentIcon = (type: Incident['type']) => {
    switch (type) {
      case 'accident': return <AlertTriangle className="w-5 h-5 text-red-500" />
      case 'congestion': return <TrendingUp className="w-5 h-5 text-orange-500" />
      case 'road_closure': return <XCircle className="w-5 h-5 text-red-500" />
      case 'weather': return <AlertCircle className="w-5 h-5 text-blue-500" />
      case 'construction': return <AlertTriangle className="w-5 h-5 text-yellow-500" />
      default: return <AlertCircle className="w-5 h-5 text-gray-500" />
    }
  }

  const handleIncidentClick = (incident: Incident) => {
    setSelectedIncident(incident)
    onIncidentSelect?.(incident)
  }

  const handleResolveIncident = (incidentId: string) => {
    onIncidentResolve?.(incidentId)
    if (selectedIncident?.id === incidentId) {
      setSelectedIncident(null)
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-1/3 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-24 bg-gray-200 rounded animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Statistics Overview */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Total Incidents</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-gray-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Active</p>
                <p className="text-2xl font-bold text-orange-600">{stats.active}</p>
              </div>
              <Activity className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Critical</p>
                <p className="text-2xl font-bold text-red-600">{stats.critical}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Resolved</p>
                <p className="text-2xl font-bold text-green-600">{stats.resolved}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Avg Resolution</p>
                <p className="text-2xl font-bold">{stats.avgResolutionTime}</p>
              </div>
              <Clock className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Incident Management Card */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Incident Management
              {isConnected && (
                <Badge variant="outline" className="ml-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse" />
                  Live
                </Badge>
              )}
            </CardTitle>

            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-1" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <RefreshCw className="w-4 h-4 mr-1" />
                Refresh
              </Button>
            </div>
          </div>

          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search incidents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <Select value={filterStatus} onValueChange={(value: any) => setFilterStatus(value)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="monitoring">Monitoring</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterSeverity} onValueChange={(value: any) => setFilterSeverity(value)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severity</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
              <SelectTrigger className="w-[120px]">
                <SelectValue placeholder="Sort" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="time">Time</SelectItem>
                <SelectItem value="severity">Severity</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>

        <CardContent>
          <div className="space-y-4">
            <AnimatePresence>
              {filteredIncidents.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <AlertCircle className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                  <p>No incidents found</p>
                </div>
              ) : (
                filteredIncidents.map((incident) => (
                  <motion.div
                    key={incident.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Alert
                      className={cn(
                        'cursor-pointer transition-all hover:shadow-md',
                        selectedIncident?.id === incident.id && 'ring-2 ring-blue-500'
                      )}
                      onClick={() => handleIncidentClick(incident)}
                    >
                      <div className="flex items-start gap-3">
                        {getIncidentIcon(incident.type)}
                        <div className="flex-1">
                          <div className="flex items-start justify-between">
                            <div>
                              <AlertTitle className="flex items-center gap-2">
                                {incident.description}
                                <Badge className={cn('ml-2', getSeverityColor(incident.severity))}>
                                  {incident.severity}
                                </Badge>
                                <Badge variant="outline" className="ml-1">
                                  {getStatusIcon(incident.status)}
                                  <span className="ml-1">{incident.status}</span>
                                </Badge>
                              </AlertTitle>
                              <AlertDescription className="mt-2">
                                <div className="flex items-center gap-4 text-sm text-gray-600">
                                  <span className="flex items-center gap-1">
                                    <Camera className="w-3 h-3" />
                                    {incident.cameraName}
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <MapPin className="w-3 h-3" />
                                    {incident.location.intersection || 'Unknown location'}
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {formatDistanceToNow(new Date(incident.timestamp), { addSuffix: true })}
                                  </span>
                                </div>
                                
                                {incident.affectedLanes && (
                                  <div className="mt-2 text-sm">
                                    <span className="font-medium">{incident.affectedLanes} lanes affected</span>
                                    {incident.vehiclesAffected && (
                                      <span className="ml-3">{incident.vehiclesAffected} vehicles impacted</span>
                                    )}
                                  </div>
                                )}

                                {incident.responders && (
                                  <div className="mt-2 flex gap-2">
                                    {incident.responders.police && <Badge variant="secondary">Police</Badge>}
                                    {incident.responders.ambulance && <Badge variant="secondary">Ambulance</Badge>}
                                    {incident.responders.fire && <Badge variant="secondary">Fire</Badge>}
                                    {incident.responders.towing && <Badge variant="secondary">Towing</Badge>}
                                  </div>
                                )}
                              </AlertDescription>
                            </div>
                            
                            <div className="flex gap-2">
                              {incident.status === 'active' && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleResolveIncident(incident.id)
                                  }}
                                >
                                  Resolve
                                </Button>
                              )}
                              <ChevronRight className="w-5 h-5 text-gray-400" />
                            </div>
                          </div>
                        </div>
                      </div>
                    </Alert>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </CardContent>
      </Card>

      {/* Selected Incident Details */}
      <AnimatePresence>
        {selectedIncident && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    Incident Details - {selectedIncident.id}
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedIncident(null)}
                  >
                    <XCircle className="w-4 h-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Type</p>
                      <p className="font-medium capitalize">{selectedIncident.type.replace('_', ' ')}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Severity</p>
                      <Badge className={getSeverityColor(selectedIncident.severity)}>
                        {selectedIncident.severity}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Started</p>
                      <p className="font-medium">
                        {format(new Date(selectedIncident.timestamp), 'MMM dd, yyyy HH:mm')}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Estimated Clear Time</p>
                      <p className="font-medium">
                        {selectedIncident.estimatedClearTime || 'Unknown'}
                      </p>
                    </div>
                  </div>

                  {selectedIncident.updates && selectedIncident.updates.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-2">Updates</h4>
                      <div className="space-y-2">
                        {selectedIncident.updates.map((update, index) => (
                          <div key={index} className="bg-gray-50 p-3 rounded">
                            <p className="text-sm">{update.message}</p>
                            <p className="text-xs text-gray-500 mt-1">
                              {format(new Date(update.timestamp), 'MMM dd, HH:mm')}
                              {update.updatedBy && ` by ${update.updatedBy}`}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2">
                    <Button className="flex-1">
                      Add Update
                    </Button>
                    {selectedIncident.status === 'active' && (
                      <Button
                        variant="outline"
                        className="flex-1"
                        onClick={() => handleResolveIncident(selectedIncident.id)}
                      >
                        Mark as Resolved
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}