'use client'

import React, { useState, useCallback, useDeferredValue, useTransition } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Progress } from '@/components/ui/progress'
import {
  IconDownload,
  IconCalendar,
  IconClock,
  IconMail,
  IconFileExport,
  IconFileTypeCsv,
  IconFileTypeXls,
  IconFileTypePdf,
  IconFileTypeDocx,
  IconReport,
  IconChartBar,
  IconTable,
  IconFilter,
  IconSettings,
  IconSend,
  IconEye,
  IconTrash,
  IconEdit,
  IconRefresh,
  IconPlus,
  IconFileText,
  IconCalendarEvent,
  IconHistory,
  IconBrandTelegram,
  IconMoodSmile
} from '@tabler/icons-react'

// Export format definitions
export type ExportFormat = 'csv' | 'excel' | 'pdf' | 'json' | 'xml'

export interface ExportRequest {
  id: string
  name: string
  description?: string
  format: ExportFormat
  dataSource: string
  dateRange: {
    start: string
    end: string
    preset?: '1h' | '24h' | '7d' | '30d' | '90d' | 'custom'
  }
  filters: Record<string, any>
  columns: string[]
  groupBy?: string[]
  aggregations?: Record<string, 'sum' | 'avg' | 'count' | 'min' | 'max'>
  includeCharts: boolean
  includeMetadata: boolean
  compression?: 'none' | 'zip' | 'gzip'
  createdAt: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  downloadUrl?: string
  errorMessage?: string
}

export interface ScheduledReport {
  id: string
  name: string
  description?: string
  exportRequest: Omit<ExportRequest, 'id' | 'status' | 'progress' | 'downloadUrl' | 'errorMessage'>
  schedule: {
    frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly'
    time: string // HH:MM format
    dayOfWeek?: number // 0-6 for weekly
    dayOfMonth?: number // 1-31 for monthly
    timezone: string
  }
  recipients: Array<{
    type: 'email' | 'webhook' | 'slack'
    address: string
    name?: string
  }>
  isActive: boolean
  lastRun?: string
  nextRun: string
  runHistory: Array<{
    timestamp: string
    status: 'success' | 'failed'
    errorMessage?: string
    downloadUrl?: string
  }>
  createdAt: string
  updatedAt: string
}

export interface ReportTemplate {
  id: string
  name: string
  description: string
  category: 'traffic' | 'camera' | 'system' | 'incidents' | 'custom'
  preview: string
  config: Omit<ExportRequest, 'id' | 'status' | 'progress' | 'downloadUrl' | 'errorMessage'>
  popularity: number
  featured: boolean
}

interface DataExportReportingProps {
  onExport?: (request: ExportRequest) => Promise<void>
  onScheduleReport?: (report: ScheduledReport) => Promise<void>
  onUpdateSchedule?: (reportId: string, updates: Partial<ScheduledReport>) => Promise<void>
  onDeleteSchedule?: (reportId: string) => Promise<void>
  availableDataSources?: string[]
  availableColumns?: Record<string, string[]>
}

// Mock report templates
const REPORT_TEMPLATES: ReportTemplate[] = [
  {
    id: 'daily-traffic',
    name: 'Daily Traffic Summary',
    description: 'Comprehensive daily traffic analysis report',
    category: 'traffic',
    preview: '/templates/daily-traffic.png',
    popularity: 98,
    featured: true,
    config: {
      name: 'Daily Traffic Summary',
      format: 'pdf',
      dataSource: 'traffic_data',
      dateRange: { start: '', end: '', preset: '24h' },
      filters: {},
      columns: ['timestamp', 'vehicleCount', 'averageSpeed', 'congestionLevel'],
      includeCharts: true,
      includeMetadata: true,
      createdAt: new Date().toISOString()
    }
  },
  {
    id: 'camera-health',
    name: 'Camera Health Report',
    description: 'Camera performance and health monitoring report',
    category: 'camera',
    preview: '/templates/camera-health.png',
    popularity: 87,
    featured: true,
    config: {
      name: 'Camera Health Report',
      format: 'excel',
      dataSource: 'camera_status',
      dateRange: { start: '', end: '', preset: '7d' },
      filters: {},
      columns: ['cameraId', 'status', 'uptime', 'frameRate', 'lastPing'],
      includeCharts: true,
      includeMetadata: true,
      createdAt: new Date().toISOString()
    }
  },
  {
    id: 'incident-analysis',
    name: 'Incident Analysis Report',
    description: 'Detailed incident analysis and response metrics',
    category: 'incidents',
    preview: '/templates/incident-analysis.png',
    popularity: 92,
    featured: false,
    config: {
      name: 'Incident Analysis Report',
      format: 'pdf',
      dataSource: 'incidents',
      dateRange: { start: '', end: '', preset: '30d' },
      filters: {},
      columns: ['id', 'type', 'severity', 'status', 'responseTime', 'resolutionTime'],
      groupBy: ['type', 'severity'],
      aggregations: { responseTime: 'avg', resolutionTime: 'avg' },
      includeCharts: true,
      includeMetadata: true,
      createdAt: new Date().toISOString()
    }
  }
]

// Mock data sources and their available columns
const DATA_SOURCES = {
  traffic_data: ['timestamp', 'cameraId', 'vehicleCount', 'averageSpeed', 'congestionLevel', 'occupancy'],
  camera_status: ['cameraId', 'status', 'uptime', 'frameRate', 'lastPing', 'location', 'model'],
  incidents: ['id', 'type', 'severity', 'status', 'timestamp', 'responseTime', 'resolutionTime', 'location'],
  alerts: ['id', 'type', 'severity', 'message', 'timestamp', 'acknowledged', 'source'],
  system_metrics: ['timestamp', 'cpuUsage', 'memoryUsage', 'diskUsage', 'networkLatency', 'apiResponseTime']
}

const getFormatIcon = (format: ExportFormat) => {
  switch (format) {
    case 'csv': return IconFileTypeCsv
    case 'excel': return IconFileTypeXls
    case 'pdf': return IconFileTypePdf
    case 'json': return IconFileText
    case 'xml': return IconFileText
    default: return IconFileExport
  }
}

export const DataExportReporting: React.FC<DataExportReportingProps> = ({
  onExport,
  onScheduleReport,
  onUpdateSchedule,
  onDeleteSchedule,
  availableDataSources = Object.keys(DATA_SOURCES),
  availableColumns = DATA_SOURCES
}) => {
  const [isPending, startTransition] = useTransition()
  const [activeTab, setActiveTab] = useState('export')

  // Export state
  const [exportRequest, setExportRequest] = useState<Partial<ExportRequest>>({
    name: 'New Export',
    format: 'csv',
    dataSource: availableDataSources[0] || 'traffic_data',
    dateRange: { start: '', end: '', preset: '24h' },
    filters: {},
    columns: [],
    includeCharts: false,
    includeMetadata: true
  })

  // Scheduled reports state
  const [scheduledReports, setScheduledReports] = useState<ScheduledReport[]>([
    {
      id: 'schedule-1',
      name: 'Daily Traffic Report',
      description: 'Automated daily traffic summary',
      exportRequest: REPORT_TEMPLATES[0].config,
      schedule: {
        frequency: 'daily',
        time: '08:00',
        timezone: 'UTC'
      },
      recipients: [
        { type: 'email', address: 'admin@its-camera.ai', name: 'Admin Team' }
      ],
      isActive: true,
      nextRun: new Date(Date.now() + 86400000).toISOString(),
      runHistory: [
        { timestamp: new Date().toISOString(), status: 'success', downloadUrl: '/exports/daily-traffic-2024-01-15.pdf' }
      ],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
  ])

  // Export history state
  const [exportHistory, setExportHistory] = useState<ExportRequest[]>([
    {
      id: 'export-1',
      name: 'Traffic Analysis - Week 47',
      format: 'excel',
      dataSource: 'traffic_data',
      dateRange: { start: '2024-01-08', end: '2024-01-14', preset: '7d' },
      filters: {},
      columns: ['timestamp', 'vehicleCount', 'averageSpeed'],
      includeCharts: true,
      includeMetadata: true,
      createdAt: new Date(Date.now() - 86400000).toISOString(),
      status: 'completed',
      progress: 100,
      downloadUrl: '/exports/traffic-analysis-week-47.xlsx'
    },
    {
      id: 'export-2',
      name: 'Camera Health Check',
      format: 'pdf',
      dataSource: 'camera_status',
      dateRange: { start: '2024-01-01', end: '2024-01-15', preset: '30d' },
      filters: {},
      columns: ['cameraId', 'status', 'uptime'],
      includeCharts: true,
      includeMetadata: true,
      createdAt: new Date(Date.now() - 172800000).toISOString(),
      status: 'processing',
      progress: 65
    }
  ])

  const [showTemplates, setShowTemplates] = useState(false)
  const [showScheduleDialog, setShowScheduleDialog] = useState(false)
  const [editingSchedule, setEditingSchedule] = useState<ScheduledReport | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState<string>('all')

  const deferredSearchQuery = useDeferredValue(searchQuery)

  // Filter templates
  const filteredTemplates = REPORT_TEMPLATES.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(deferredSearchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(deferredSearchQuery.toLowerCase())
    const matchesCategory = filterCategory === 'all' || template.category === filterCategory
    return matchesSearch && matchesCategory
  })

  const handleExport = useCallback(async () => {
    if (!exportRequest.name || !exportRequest.dataSource) return

    const request: ExportRequest = {
      id: crypto.randomUUID(),
      name: exportRequest.name,
      description: exportRequest.description,
      format: exportRequest.format || 'csv',
      dataSource: exportRequest.dataSource,
      dateRange: exportRequest.dateRange || { start: '', end: '', preset: '24h' },
      filters: exportRequest.filters || {},
      columns: exportRequest.columns || [],
      groupBy: exportRequest.groupBy,
      aggregations: exportRequest.aggregations,
      includeCharts: exportRequest.includeCharts || false,
      includeMetadata: exportRequest.includeMetadata || true,
      compression: exportRequest.compression,
      createdAt: new Date().toISOString(),
      status: 'pending',
      progress: 0
    }

    startTransition(() => {
      setExportHistory(prev => [request, ...prev])
    })

    // Simulate export process
    setTimeout(() => {
      setExportHistory(prev => prev.map(exp =>
        exp.id === request.id
          ? { ...exp, status: 'processing', progress: 25 }
          : exp
      ))
    }, 1000)

    setTimeout(() => {
      setExportHistory(prev => prev.map(exp =>
        exp.id === request.id
          ? { ...exp, status: 'processing', progress: 75 }
          : exp
      ))
    }, 3000)

    setTimeout(() => {
      setExportHistory(prev => prev.map(exp =>
        exp.id === request.id
          ? {
              ...exp,
              status: 'completed',
              progress: 100,
              downloadUrl: `/exports/${request.name.toLowerCase().replace(/\s+/g, '-')}.${request.format}`
            }
          : exp
      ))
    }, 5000)

    await onExport?.(request)
  }, [exportRequest, onExport])

  const loadTemplate = useCallback((template: ReportTemplate) => {
    setExportRequest({
      ...template.config,
      dateRange: {
        ...template.config.dateRange,
        start: new Date(Date.now() - 86400000).toISOString().split('T')[0],
        end: new Date().toISOString().split('T')[0]
      }
    })
    setShowTemplates(false)
  }, [])

  const handleScheduleReport = useCallback(async () => {
    if (!editingSchedule) return

    if (editingSchedule.id.startsWith('new-')) {
      // Create new schedule
      const newSchedule: ScheduledReport = {
        ...editingSchedule,
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      setScheduledReports(prev => [...prev, newSchedule])
      await onScheduleReport?.(newSchedule)
    } else {
      // Update existing schedule
      setScheduledReports(prev => prev.map(schedule =>
        schedule.id === editingSchedule.id ? { ...editingSchedule, updatedAt: new Date().toISOString() } : schedule
      ))
      await onUpdateSchedule?.(editingSchedule.id, editingSchedule)
    }

    setEditingSchedule(null)
    setShowScheduleDialog(false)
  }, [editingSchedule, onScheduleReport, onUpdateSchedule])

  const createNewSchedule = useCallback(() => {
    setEditingSchedule({
      id: 'new-' + crypto.randomUUID(),
      name: 'New Scheduled Report',
      exportRequest: {
        name: 'Scheduled Report',
        format: 'pdf',
        dataSource: availableDataSources[0] || 'traffic_data',
        dateRange: { start: '', end: '', preset: '24h' },
        filters: {},
        columns: [],
        includeCharts: true,
        includeMetadata: true,
        createdAt: new Date().toISOString()
      },
      schedule: {
        frequency: 'daily',
        time: '08:00',
        timezone: 'UTC'
      },
      recipients: [],
      isActive: true,
      nextRun: new Date(Date.now() + 86400000).toISOString(),
      runHistory: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    })
    setShowScheduleDialog(true)
  }, [availableDataSources])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Data Export & Reporting</h2>
          <p className="text-muted-foreground">
            Export data and create automated reports
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            onClick={() => setShowTemplates(true)}
          >
            <IconReport className="h-4 w-4 mr-2" />
            Templates
          </Button>
          <Button onClick={createNewSchedule}>
            <IconPlus className="h-4 w-4 mr-2" />
            Schedule Report
          </Button>
        </div>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="export">Quick Export</TabsTrigger>
          <TabsTrigger value="scheduled">Scheduled Reports</TabsTrigger>
          <TabsTrigger value="history">Export History</TabsTrigger>
        </TabsList>

        {/* Quick Export Tab */}
        <TabsContent value="export" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Export Configuration */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Export Configuration</CardTitle>
                  <CardDescription>Configure your data export settings</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="export-name">Export Name</Label>
                      <Input
                        id="export-name"
                        value={exportRequest.name || ''}
                        onChange={(e) => setExportRequest(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="My Export"
                      />
                    </div>
                    <div>
                      <Label htmlFor="export-format">Format</Label>
                      <Select
                        value={exportRequest.format}
                        onValueChange={(value: ExportFormat) => setExportRequest(prev => ({ ...prev, format: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="csv">CSV</SelectItem>
                          <SelectItem value="excel">Excel</SelectItem>
                          <SelectItem value="pdf">PDF</SelectItem>
                          <SelectItem value="json">JSON</SelectItem>
                          <SelectItem value="xml">XML</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="data-source">Data Source</Label>
                    <Select
                      value={exportRequest.dataSource}
                      onValueChange={(value) => setExportRequest(prev => ({
                        ...prev,
                        dataSource: value,
                        columns: [] // Reset columns when data source changes
                      }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableDataSources.map(source => (
                          <SelectItem key={source} value={source}>
                            {source.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="export-description">Description (Optional)</Label>
                    <Textarea
                      id="export-description"
                      value={exportRequest.description || ''}
                      onChange={(e) => setExportRequest(prev => ({ ...prev, description: e.target.value }))}
                      placeholder="Describe this export..."
                      rows={2}
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="date-preset">Date Range</Label>
                      <Select
                        value={exportRequest.dateRange?.preset || '24h'}
                        onValueChange={(value) => setExportRequest(prev => ({
                          ...prev,
                          dateRange: { ...prev.dateRange, preset: value as any }
                        }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1h">Last Hour</SelectItem>
                          <SelectItem value="24h">Last 24 Hours</SelectItem>
                          <SelectItem value="7d">Last 7 Days</SelectItem>
                          <SelectItem value="30d">Last 30 Days</SelectItem>
                          <SelectItem value="90d">Last 90 Days</SelectItem>
                          <SelectItem value="custom">Custom Range</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="compression">Compression</Label>
                      <Select
                        value={exportRequest.compression || 'none'}
                        onValueChange={(value) => setExportRequest(prev => ({ ...prev, compression: value as any }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="zip">ZIP</SelectItem>
                          <SelectItem value="gzip">GZIP</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {exportRequest.dateRange?.preset === 'custom' && (
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="start-date">Start Date</Label>
                        <Input
                          id="start-date"
                          type="date"
                          value={exportRequest.dateRange?.start || ''}
                          onChange={(e) => setExportRequest(prev => ({
                            ...prev,
                            dateRange: { ...prev.dateRange, start: e.target.value }
                          }))}
                        />
                      </div>
                      <div>
                        <Label htmlFor="end-date">End Date</Label>
                        <Input
                          id="end-date"
                          type="date"
                          value={exportRequest.dateRange?.end || ''}
                          onChange={(e) => setExportRequest(prev => ({
                            ...prev,
                            dateRange: { ...prev.dateRange, end: e.target.value }
                          }))}
                        />
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Column Selection */}
              <Card>
                <CardHeader>
                  <CardTitle>Columns</CardTitle>
                  <CardDescription>Select which columns to include in the export</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {exportRequest.dataSource && availableColumns[exportRequest.dataSource]?.map(column => (
                      <div key={column} className="flex items-center space-x-2">
                        <Checkbox
                          id={`column-${column}`}
                          checked={exportRequest.columns?.includes(column) || false}
                          onCheckedChange={(checked) => {
                            setExportRequest(prev => ({
                              ...prev,
                              columns: checked
                                ? [...(prev.columns || []), column]
                                : (prev.columns || []).filter(c => c !== column)
                            }))
                          }}
                        />
                        <Label htmlFor={`column-${column}`} className="text-sm">
                          {column.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                        </Label>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Export Options */}
              <Card>
                <CardHeader>
                  <CardTitle>Export Options</CardTitle>
                  <CardDescription>Additional export settings</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="include-charts">Include Charts</Label>
                      <p className="text-sm text-muted-foreground">Add visualizations to the export</p>
                    </div>
                    <Switch
                      id="include-charts"
                      checked={exportRequest.includeCharts || false}
                      onCheckedChange={(checked) => setExportRequest(prev => ({ ...prev, includeCharts: checked }))}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="include-metadata">Include Metadata</Label>
                      <p className="text-sm text-muted-foreground">Add export details and timestamps</p>
                    </div>
                    <Switch
                      id="include-metadata"
                      checked={exportRequest.includeMetadata || false}
                      onCheckedChange={(checked) => setExportRequest(prev => ({ ...prev, includeMetadata: checked }))}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Export Preview & Actions */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Export Preview</CardTitle>
                  <CardDescription>Summary of your export configuration</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Name:</span>
                      <span className="text-sm font-medium">{exportRequest.name || 'Untitled'}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Format:</span>
                      <div className="flex items-center space-x-1">
                        {exportRequest.format && React.createElement(getFormatIcon(exportRequest.format), {
                          className: "h-4 w-4"
                        })}
                        <span className="text-sm font-medium">{exportRequest.format?.toUpperCase()}</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Data Source:</span>
                      <span className="text-sm font-medium">
                        {exportRequest.dataSource?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Columns:</span>
                      <span className="text-sm font-medium">{exportRequest.columns?.length || 0} selected</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Date Range:</span>
                      <span className="text-sm font-medium">
                        {exportRequest.dateRange?.preset === 'custom'
                          ? `${exportRequest.dateRange.start} to ${exportRequest.dateRange.end}`
                          : exportRequest.dateRange?.preset?.toUpperCase()
                        }
                      </span>
                    </div>
                  </div>

                  <div className="pt-4 border-t">
                    <Button
                      className="w-full"
                      onClick={handleExport}
                      disabled={!exportRequest.name || !exportRequest.dataSource || !exportRequest.columns?.length || isPending}
                    >
                      <IconDownload className="h-4 w-4 mr-2" />
                      Export Data
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                  <CardDescription>Common export templates</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  {REPORT_TEMPLATES.slice(0, 3).map((template) => (
                    <Button
                      key={template.id}
                      variant="outline"
                      className="w-full justify-start"
                      onClick={() => loadTemplate(template)}
                    >
                      <IconFileExport className="h-4 w-4 mr-2" />
                      {template.name}
                    </Button>
                  ))}
                  <Button
                    variant="ghost"
                    className="w-full"
                    onClick={() => setShowTemplates(true)}
                  >
                    View All Templates
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Scheduled Reports Tab */}
        <TabsContent value="scheduled" className="space-y-6">
          <div className="grid gap-4">
            {scheduledReports.map((report) => (
              <Card key={report.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">{report.name}</CardTitle>
                      <CardDescription>{report.description}</CardDescription>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={report.isActive ? "default" : "secondary"}>
                        {report.isActive ? "Active" : "Inactive"}
                      </Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setEditingSchedule(report)
                          setShowScheduleDialog(true)
                        }}
                      >
                        <IconEdit className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onDeleteSchedule?.(report.id)}
                      >
                        <IconTrash className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="text-sm font-medium mb-2">Schedule</h4>
                      <div className="space-y-1 text-sm text-muted-foreground">
                        <div className="flex items-center space-x-2">
                          <IconCalendarEvent className="h-4 w-4" />
                          <span>{report.schedule.frequency} at {report.schedule.time}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <IconClock className="h-4 w-4" />
                          <span>Next run: {new Date(report.nextRun).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-2">Recipients</h4>
                      <div className="space-y-1">
                        {report.recipients.map((recipient, index) => (
                          <div key={index} className="flex items-center space-x-2 text-sm text-muted-foreground">
                            <IconMail className="h-4 w-4" />
                            <span>{recipient.name || recipient.address}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-2">Last Run</h4>
                      {report.runHistory.length > 0 ? (
                        <div className="space-y-1 text-sm text-muted-foreground">
                          <div className="flex items-center space-x-2">
                            <IconHistory className="h-4 w-4" />
                            <span>{new Date(report.runHistory[0].timestamp).toLocaleString()}</span>
                          </div>
                          <Badge variant={report.runHistory[0].status === 'success' ? 'default' : 'destructive'}>
                            {report.runHistory[0].status}
                          </Badge>
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">Never run</span>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Export History Tab */}
        <TabsContent value="history" className="space-y-6">
          <div className="grid gap-4">
            {exportHistory.map((export_) => (
              <Card key={export_.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">{export_.name}</CardTitle>
                      <CardDescription>
                        {export_.format.toUpperCase()} • {export_.dataSource.replace('_', ' ')} • {new Date(export_.createdAt).toLocaleString()}
                      </CardDescription>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge
                        variant={
                          export_.status === 'completed' ? 'default' :
                          export_.status === 'processing' ? 'secondary' :
                          export_.status === 'failed' ? 'destructive' : 'outline'
                        }
                      >
                        {export_.status}
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {export_.status === 'processing' && (
                      <div>
                        <div className="flex items-center justify-between text-sm mb-2">
                          <span>Progress</span>
                          <span>{export_.progress}%</span>
                        </div>
                        <Progress value={export_.progress} />
                      </div>
                    )}

                    {export_.status === 'failed' && export_.errorMessage && (
                      <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
                        <p className="text-sm text-destructive">{export_.errorMessage}</p>
                      </div>
                    )}

                    <div className="flex items-center justify-between">
                      <div className="text-sm text-muted-foreground">
                        {export_.columns.length} columns • {export_.includeCharts ? 'With charts' : 'Data only'}
                      </div>
                      {export_.status === 'completed' && export_.downloadUrl && (
                        <Button size="sm" variant="outline">
                          <IconDownload className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* Template Selection Dialog */}
      <Dialog open={showTemplates} onOpenChange={setShowTemplates}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Export Templates</DialogTitle>
            <DialogDescription>
              Choose from pre-configured export templates
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1"
              />
              <Select value={filterCategory} onValueChange={setFilterCategory}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Categories</SelectItem>
                  <SelectItem value="traffic">Traffic</SelectItem>
                  <SelectItem value="camera">Camera</SelectItem>
                  <SelectItem value="incidents">Incidents</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredTemplates.map((template) => (
                <Card key={template.id} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">{template.name}</CardTitle>
                      {template.featured && (
                        <Badge variant="secondary">Featured</Badge>
                      )}
                    </div>
                    <CardDescription>{template.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {template.category}
                          </Badge>
                          <span className="text-muted-foreground">
                            {template.config.format.toUpperCase()}
                          </span>
                        </div>
                        <span className="text-muted-foreground">
                          {template.popularity}% popularity
                        </span>
                      </div>

                      <Button
                        className="w-full"
                        onClick={() => loadTemplate(template)}
                      >
                        Use Template
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Schedule Report Dialog */}
      <Dialog open={showScheduleDialog} onOpenChange={setShowScheduleDialog}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingSchedule?.id.startsWith('new-') ? 'Create Scheduled Report' : 'Edit Scheduled Report'}
            </DialogTitle>
            <DialogDescription>
              Configure automated report generation and delivery
            </DialogDescription>
          </DialogHeader>

          {editingSchedule && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="schedule-name">Report Name</Label>
                  <Input
                    id="schedule-name"
                    value={editingSchedule.name}
                    onChange={(e) => setEditingSchedule(prev => prev ? { ...prev, name: e.target.value } : null)}
                  />
                </div>
                <div>
                  <Label htmlFor="schedule-frequency">Frequency</Label>
                  <Select
                    value={editingSchedule.schedule.frequency}
                    onValueChange={(value: any) => setEditingSchedule(prev => prev ? {
                      ...prev,
                      schedule: { ...prev.schedule, frequency: value }
                    } : null)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                      <SelectItem value="quarterly">Quarterly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="schedule-description">Description</Label>
                <Textarea
                  id="schedule-description"
                  value={editingSchedule.description || ''}
                  onChange={(e) => setEditingSchedule(prev => prev ? { ...prev, description: e.target.value } : null)}
                  rows={2}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="schedule-time">Time</Label>
                  <Input
                    id="schedule-time"
                    type="time"
                    value={editingSchedule.schedule.time}
                    onChange={(e) => setEditingSchedule(prev => prev ? {
                      ...prev,
                      schedule: { ...prev.schedule, time: e.target.value }
                    } : null)}
                  />
                </div>
                <div>
                  <Label htmlFor="schedule-timezone">Timezone</Label>
                  <Select
                    value={editingSchedule.schedule.timezone}
                    onValueChange={(value) => setEditingSchedule(prev => prev ? {
                      ...prev,
                      schedule: { ...prev.schedule, timezone: value }
                    } : null)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="UTC">UTC</SelectItem>
                      <SelectItem value="America/New_York">Eastern Time</SelectItem>
                      <SelectItem value="America/Los_Angeles">Pacific Time</SelectItem>
                      <SelectItem value="Europe/London">London</SelectItem>
                      <SelectItem value="Asia/Tokyo">Tokyo</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label>Recipients</Label>
                <div className="space-y-2 mt-2">
                  {editingSchedule.recipients.map((recipient, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Select
                        value={recipient.type}
                        onValueChange={(value: any) => {
                          const newRecipients = [...editingSchedule.recipients]
                          newRecipients[index] = { ...recipient, type: value }
                          setEditingSchedule(prev => prev ? { ...prev, recipients: newRecipients } : null)
                        }}
                      >
                        <SelectTrigger className="w-32">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="email">Email</SelectItem>
                          <SelectItem value="webhook">Webhook</SelectItem>
                          <SelectItem value="slack">Slack</SelectItem>
                        </SelectContent>
                      </Select>
                      <Input
                        placeholder="Address"
                        value={recipient.address}
                        onChange={(e) => {
                          const newRecipients = [...editingSchedule.recipients]
                          newRecipients[index] = { ...recipient, address: e.target.value }
                          setEditingSchedule(prev => prev ? { ...prev, recipients: newRecipients } : null)
                        }}
                        className="flex-1"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const newRecipients = editingSchedule.recipients.filter((_, i) => i !== index)
                          setEditingSchedule(prev => prev ? { ...prev, recipients: newRecipients } : null)
                        }}
                      >
                        <IconTrash className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const newRecipients = [...editingSchedule.recipients, { type: 'email' as const, address: '' }]
                      setEditingSchedule(prev => prev ? { ...prev, recipients: newRecipients } : null)
                    }}
                  >
                    <IconPlus className="h-4 w-4 mr-2" />
                    Add Recipient
                  </Button>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="schedule-active">Active</Label>
                <Switch
                  id="schedule-active"
                  checked={editingSchedule.isActive}
                  onCheckedChange={(checked) => setEditingSchedule(prev => prev ? { ...prev, isActive: checked } : null)}
                />
              </div>

              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setShowScheduleDialog(false)}>
                  Cancel
                </Button>
                <Button onClick={handleScheduleReport}>
                  <IconSave className="h-4 w-4 mr-2" />
                  Save Schedule
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default DataExportReporting
