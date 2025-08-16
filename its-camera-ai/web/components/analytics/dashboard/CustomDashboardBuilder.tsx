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
import {
  IconPlus,
  IconGripVertical,
  IconTrash,
  IconEdit,
  IconCopy,
  IconSave,
  IconDownload,
  IconUpload,
  IconGrid3x3,
  IconLayout,
  IconChartBar,
  IconChartLine,
  IconChartPie,
  IconTable,
  IconMap,
  IconCamera,
  IconAlertTriangle,
  IconSettings,
  IconEye,
  IconDeviceFloppy,
  IconTemplate,
  IconPalette,
  IconResize
} from '@tabler/icons-react'

// Widget type definitions
export interface DashboardWidget {
  id: string
  type: 'chart' | 'metric' | 'table' | 'map' | 'alert' | 'camera'
  title: string
  description?: string
  position: { x: number; y: number }
  size: { width: number; height: number }
  config: {
    dataSource?: string
    chartType?: 'line' | 'bar' | 'pie' | 'area'
    refreshInterval?: number
    showLegend?: boolean
    showGrid?: boolean
    filters?: Record<string, any>
    [key: string]: any
  }
  style: {
    backgroundColor?: string
    borderColor?: string
    borderWidth?: number
    borderRadius?: number
    padding?: number
    titleColor?: string
    textColor?: string
  }
}

export interface DashboardLayout {
  id: string
  name: string
  description?: string
  widgets: DashboardWidget[]
  gridSize: { columns: number; rows: number }
  settings: {
    autoRefresh: boolean
    refreshInterval: number
    theme: 'light' | 'dark' | 'auto'
    showGrid: boolean
    snapToGrid: boolean
  }
  createdAt: string
  updatedAt: string
  isTemplate: boolean
  tags: string[]
}

export interface DashboardTemplate {
  id: string
  name: string
  description: string
  category: 'traffic' | 'camera' | 'system' | 'custom'
  preview: string
  layout: Omit<DashboardLayout, 'id' | 'createdAt' | 'updatedAt' | 'isTemplate'>
  popularity: number
  featured: boolean
}

interface CustomDashboardBuilderProps {
  onSave?: (layout: DashboardLayout) => void
  onLoad?: (layoutId: string) => Promise<DashboardLayout>
  onExport?: (layout: DashboardLayout) => void
  onImport?: (file: File) => Promise<DashboardLayout>
  availableDataSources?: string[]
  initialLayout?: DashboardLayout
}

// Mock widget library
const WIDGET_LIBRARY: Omit<DashboardWidget, 'id' | 'position'>[] = [
  {
    type: 'chart',
    title: 'Traffic Flow Chart',
    description: 'Real-time vehicle count and speed trends',
    size: { width: 400, height: 300 },
    config: {
      dataSource: 'traffic_data',
      chartType: 'line',
      refreshInterval: 5000,
      showLegend: true,
      showGrid: true
    },
    style: {
      backgroundColor: 'hsl(var(--card))',
      borderColor: 'hsl(var(--border))',
      borderWidth: 1,
      borderRadius: 8,
      padding: 16
    }
  },
  {
    type: 'metric',
    title: 'Active Cameras',
    description: 'Number of operational cameras',
    size: { width: 200, height: 150 },
    config: {
      dataSource: 'camera_status',
      refreshInterval: 10000,
      format: 'number',
      showTrend: true
    },
    style: {
      backgroundColor: 'hsl(var(--card))',
      borderColor: 'hsl(var(--border))',
      borderWidth: 1,
      borderRadius: 8,
      padding: 16
    }
  },
  {
    type: 'table',
    title: 'Recent Incidents',
    description: 'Latest traffic incidents and alerts',
    size: { width: 500, height: 300 },
    config: {
      dataSource: 'incidents',
      refreshInterval: 15000,
      pageSize: 10,
      sortable: true
    },
    style: {
      backgroundColor: 'hsl(var(--card))',
      borderColor: 'hsl(var(--border))',
      borderWidth: 1,
      borderRadius: 8,
      padding: 16
    }
  },
  {
    type: 'map',
    title: 'Traffic Heatmap',
    description: 'Geographic traffic density visualization',
    size: { width: 600, height: 400 },
    config: {
      dataSource: 'heatmap_data',
      refreshInterval: 30000,
      mapType: 'traffic',
      showCameras: true
    },
    style: {
      backgroundColor: 'hsl(var(--card))',
      borderColor: 'hsl(var(--border))',
      borderWidth: 1,
      borderRadius: 8,
      padding: 16
    }
  },
  {
    type: 'alert',
    title: 'System Alerts',
    description: 'Critical system notifications',
    size: { width: 300, height: 200 },
    config: {
      dataSource: 'alerts',
      refreshInterval: 5000,
      severity: ['high', 'critical'],
      autoAck: false
    },
    style: {
      backgroundColor: 'hsl(var(--card))',
      borderColor: 'hsl(var(--destructive))',
      borderWidth: 2,
      borderRadius: 8,
      padding: 16
    }
  },
  {
    type: 'camera',
    title: 'Camera Feed',
    description: 'Live camera stream preview',
    size: { width: 320, height: 240 },
    config: {
      dataSource: 'camera_feed',
      refreshInterval: 1000,
      cameraId: 'CAM001',
      showControls: true
    },
    style: {
      backgroundColor: 'hsl(var(--card))',
      borderColor: 'hsl(var(--border))',
      borderWidth: 1,
      borderRadius: 8,
      padding: 8
    }
  }
]

// Mock dashboard templates
const DASHBOARD_TEMPLATES: DashboardTemplate[] = [
  {
    id: 'traffic-overview',
    name: 'Traffic Overview',
    description: 'Comprehensive traffic monitoring dashboard',
    category: 'traffic',
    preview: '/templates/traffic-overview.png',
    popularity: 95,
    featured: true,
    layout: {
      name: 'Traffic Overview Template',
      description: 'Pre-configured dashboard for traffic monitoring',
      widgets: [
        { ...WIDGET_LIBRARY[0], id: 'w1', position: { x: 0, y: 0 } },
        { ...WIDGET_LIBRARY[1], id: 'w2', position: { x: 400, y: 0 } },
        { ...WIDGET_LIBRARY[3], id: 'w3', position: { x: 0, y: 300 } }
      ],
      gridSize: { columns: 12, rows: 8 },
      settings: {
        autoRefresh: true,
        refreshInterval: 30000,
        theme: 'auto',
        showGrid: true,
        snapToGrid: true
      },
      tags: ['traffic', 'monitoring', 'overview']
    }
  },
  {
    id: 'camera-monitoring',
    name: 'Camera Monitoring',
    description: 'Camera health and performance dashboard',
    category: 'camera',
    preview: '/templates/camera-monitoring.png',
    popularity: 87,
    featured: true,
    layout: {
      name: 'Camera Monitoring Template',
      description: 'Monitor camera status and performance',
      widgets: [
        { ...WIDGET_LIBRARY[1], id: 'w1', position: { x: 0, y: 0 } },
        { ...WIDGET_LIBRARY[5], id: 'w2', position: { x: 200, y: 0 } },
        { ...WIDGET_LIBRARY[4], id: 'w3', position: { x: 0, y: 250 } }
      ],
      gridSize: { columns: 12, rows: 8 },
      settings: {
        autoRefresh: true,
        refreshInterval: 15000,
        theme: 'auto',
        showGrid: true,
        snapToGrid: true
      },
      tags: ['camera', 'monitoring', 'health']
    }
  },
  {
    id: 'incident-response',
    name: 'Incident Response',
    description: 'Real-time incident management dashboard',
    category: 'system',
    preview: '/templates/incident-response.png',
    popularity: 92,
    featured: false,
    layout: {
      name: 'Incident Response Template',
      description: 'Track and manage traffic incidents',
      widgets: [
        { ...WIDGET_LIBRARY[2], id: 'w1', position: { x: 0, y: 0 } },
        { ...WIDGET_LIBRARY[4], id: 'w2', position: { x: 500, y: 0 } },
        { ...WIDGET_LIBRARY[3], id: 'w3', position: { x: 0, y: 300 } }
      ],
      gridSize: { columns: 12, rows: 8 },
      settings: {
        autoRefresh: true,
        refreshInterval: 5000,
        theme: 'auto',
        showGrid: true,
        snapToGrid: true
      },
      tags: ['incidents', 'response', 'alerts']
    }
  }
]

const getWidgetIcon = (type: string) => {
  switch (type) {
    case 'chart': return IconChartLine
    case 'metric': return IconChartBar
    case 'table': return IconTable
    case 'map': return IconMap
    case 'alert': return IconAlertTriangle
    case 'camera': return IconCamera
    default: return IconGrid3x3
  }
}

export const CustomDashboardBuilder: React.FC<CustomDashboardBuilderProps> = ({
  onSave,
  onLoad,
  onExport,
  onImport,
  availableDataSources = ['traffic_data', 'camera_status', 'incidents', 'alerts'],
  initialLayout
}) => {
  const [isPending, startTransition] = useTransition()
  const [currentLayout, setCurrentLayout] = useState<DashboardLayout>(
    initialLayout || {
      id: crypto.randomUUID(),
      name: 'New Dashboard',
      description: '',
      widgets: [],
      gridSize: { columns: 12, rows: 8 },
      settings: {
        autoRefresh: true,
        refreshInterval: 30000,
        theme: 'auto',
        showGrid: true,
        snapToGrid: true
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isTemplate: false,
      tags: []
    }
  )

  const [selectedWidget, setSelectedWidget] = useState<DashboardWidget | null>(null)
  const [draggedWidget, setDraggedWidget] = useState<DashboardWidget | null>(null)
  const [showWidgetConfig, setShowWidgetConfig] = useState(false)
  const [showLayoutSettings, setShowLayoutSettings] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState<string>('all')

  const deferredSearchQuery = useDeferredValue(searchQuery)

  // Filter templates based on search and category
  const filteredTemplates = DASHBOARD_TEMPLATES.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(deferredSearchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(deferredSearchQuery.toLowerCase())
    const matchesCategory = filterCategory === 'all' || template.category === filterCategory
    return matchesSearch && matchesCategory
  })

  const addWidget = useCallback((widgetTemplate: Omit<DashboardWidget, 'id' | 'position'>) => {
    startTransition(() => {
      const newWidget: DashboardWidget = {
        ...widgetTemplate,
        id: crypto.randomUUID(),
        position: { x: 0, y: 0 }
      }

      setCurrentLayout(prev => ({
        ...prev,
        widgets: [...prev.widgets, newWidget],
        updatedAt: new Date().toISOString()
      }))
    })
  }, [])

  const updateWidget = useCallback((widgetId: string, updates: Partial<DashboardWidget>) => {
    setCurrentLayout(prev => ({
      ...prev,
      widgets: prev.widgets.map(widget =>
        widget.id === widgetId ? { ...widget, ...updates } : widget
      ),
      updatedAt: new Date().toISOString()
    }))
  }, [])

  const deleteWidget = useCallback((widgetId: string) => {
    setCurrentLayout(prev => ({
      ...prev,
      widgets: prev.widgets.filter(widget => widget.id !== widgetId),
      updatedAt: new Date().toISOString()
    }))
  }, [])

  const duplicateWidget = useCallback((widget: DashboardWidget) => {
    const newWidget: DashboardWidget = {
      ...widget,
      id: crypto.randomUUID(),
      title: `${widget.title} (Copy)`,
      position: { x: widget.position.x + 20, y: widget.position.y + 20 }
    }

    setCurrentLayout(prev => ({
      ...prev,
      widgets: [...prev.widgets, newWidget],
      updatedAt: new Date().toISOString()
    }))
  }, [])

  const loadTemplate = useCallback((template: DashboardTemplate) => {
    startTransition(() => {
      setCurrentLayout({
        ...template.layout,
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        isTemplate: false
      })
      setShowTemplates(false)
    })
  }, [])

  const handleSave = useCallback(() => {
    onSave?.(currentLayout)
  }, [currentLayout, onSave])

  const handleExport = useCallback(() => {
    onExport?.(currentLayout)
  }, [currentLayout, onExport])

  return (
    <div className="flex h-screen">
      {/* Left Sidebar - Widget Library */}
      <div className="w-80 border-r bg-muted/10 p-4 overflow-y-auto">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Widget Library</h3>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowTemplates(true)}
            >
              <IconTemplate className="h-4 w-4 mr-2" />
              Templates
            </Button>
          </div>

          <div className="space-y-2">
            {WIDGET_LIBRARY.map((widget, index) => {
              const IconComponent = getWidgetIcon(widget.type)
              return (
                <Card
                  key={index}
                  className="cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => addWidget(widget)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-start space-x-3">
                      <IconComponent className="h-5 w-5 text-primary mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium truncate">
                          {widget.title}
                        </h4>
                        <p className="text-xs text-muted-foreground line-clamp-2">
                          {widget.description}
                        </p>
                        <div className="flex items-center space-x-2 mt-2">
                          <Badge variant="secondary" className="text-xs">
                            {widget.type}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {widget.size.width}×{widget.size.height}
                          </span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Toolbar */}
        <div className="border-b p-4 bg-background">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div>
                <h2 className="text-xl font-semibold">{currentLayout.name}</h2>
                <p className="text-sm text-muted-foreground">
                  {currentLayout.widgets.length} widgets • Last modified {new Date(currentLayout.updatedAt).toLocaleTimeString()}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowLayoutSettings(true)}
              >
                <IconSettings className="h-4 w-4 mr-2" />
                Settings
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleExport}
              >
                <IconDownload className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={isPending}
              >
                <IconSave className="h-4 w-4 mr-2" />
                Save
              </Button>
            </div>
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 p-4 bg-muted/5 overflow-auto">
          <div
            className="relative min-h-full"
            style={{
              backgroundImage: currentLayout.settings.showGrid
                ? 'radial-gradient(circle, hsl(var(--muted)) 1px, transparent 1px)'
                : 'none',
              backgroundSize: currentLayout.settings.showGrid ? '20px 20px' : 'none'
            }}
          >
            {currentLayout.widgets.map((widget) => (
              <div
                key={widget.id}
                className="absolute border rounded-lg bg-card shadow-sm hover:shadow-md transition-shadow cursor-move group"
                style={{
                  left: widget.position.x,
                  top: widget.position.y,
                  width: widget.size.width,
                  height: widget.size.height,
                  backgroundColor: widget.style.backgroundColor,
                  borderColor: widget.style.borderColor,
                  borderWidth: widget.style.borderWidth,
                  borderRadius: widget.style.borderRadius,
                  padding: widget.style.padding
                }}
                onClick={() => setSelectedWidget(widget)}
              >
                {/* Widget Header */}
                <div className="flex items-center justify-between p-2 border-b">
                  <div className="flex items-center space-x-2">
                    <IconGripVertical className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium truncate">
                      {widget.title}
                    </span>
                  </div>

                  <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedWidget(widget)
                        setShowWidgetConfig(true)
                      }}
                    >
                      <IconEdit className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      onClick={(e) => {
                        e.stopPropagation()
                        duplicateWidget(widget)
                      }}
                    >
                      <IconCopy className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteWidget(widget.id)
                      }}
                    >
                      <IconTrash className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                {/* Widget Content Preview */}
                <div className="p-4 flex items-center justify-center text-muted-foreground text-sm">
                  {widget.type === 'chart' && <IconChartLine className="h-8 w-8" />}
                  {widget.type === 'metric' && <IconChartBar className="h-8 w-8" />}
                  {widget.type === 'table' && <IconTable className="h-8 w-8" />}
                  {widget.type === 'map' && <IconMap className="h-8 w-8" />}
                  {widget.type === 'alert' && <IconAlertTriangle className="h-8 w-8" />}
                  {widget.type === 'camera' && <IconCamera className="h-8 w-8" />}
                </div>

                {/* Resize Handle */}
                <div className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize opacity-0 group-hover:opacity-100 transition-opacity">
                  <IconResize className="h-3 w-3 text-muted-foreground" />
                </div>
              </div>
            ))}

            {/* Empty State */}
            {currentLayout.widgets.length === 0 && (
              <div className="flex flex-col items-center justify-center h-96 text-center">
                <IconGrid3x3 className="h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">Empty Dashboard</h3>
                <p className="text-muted-foreground mb-4">
                  Add widgets from the library to start building your dashboard
                </p>
                <Button onClick={() => setShowTemplates(true)}>
                  <IconTemplate className="h-4 w-4 mr-2" />
                  Browse Templates
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right Sidebar - Properties Panel */}
      <div className="w-80 border-l bg-muted/10 p-4 overflow-y-auto">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Properties</h3>

          {selectedWidget ? (
            <div className="space-y-4">
              <div>
                <Label htmlFor="widget-title">Title</Label>
                <Input
                  id="widget-title"
                  value={selectedWidget.title}
                  onChange={(e) => updateWidget(selectedWidget.id, { title: e.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="widget-description">Description</Label>
                <Textarea
                  id="widget-description"
                  value={selectedWidget.description || ''}
                  onChange={(e) => updateWidget(selectedWidget.id, { description: e.target.value })}
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label htmlFor="widget-width">Width</Label>
                  <Input
                    id="widget-width"
                    type="number"
                    value={selectedWidget.size.width}
                    onChange={(e) => updateWidget(selectedWidget.id, {
                      size: { ...selectedWidget.size, width: parseInt(e.target.value) }
                    })}
                  />
                </div>
                <div>
                  <Label htmlFor="widget-height">Height</Label>
                  <Input
                    id="widget-height"
                    type="number"
                    value={selectedWidget.size.height}
                    onChange={(e) => updateWidget(selectedWidget.id, {
                      size: { ...selectedWidget.size, height: parseInt(e.target.value) }
                    })}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label htmlFor="widget-x">X Position</Label>
                  <Input
                    id="widget-x"
                    type="number"
                    value={selectedWidget.position.x}
                    onChange={(e) => updateWidget(selectedWidget.id, {
                      position: { ...selectedWidget.position, x: parseInt(e.target.value) }
                    })}
                  />
                </div>
                <div>
                  <Label htmlFor="widget-y">Y Position</Label>
                  <Input
                    id="widget-y"
                    type="number"
                    value={selectedWidget.position.y}
                    onChange={(e) => updateWidget(selectedWidget.id, {
                      position: { ...selectedWidget.position, y: parseInt(e.target.value) }
                    })}
                  />
                </div>
              </div>

              {selectedWidget.type === 'chart' && (
                <div>
                  <Label htmlFor="chart-type">Chart Type</Label>
                  <Select
                    value={selectedWidget.config.chartType}
                    onValueChange={(value) => updateWidget(selectedWidget.id, {
                      config: { ...selectedWidget.config, chartType: value }
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="line">Line Chart</SelectItem>
                      <SelectItem value="bar">Bar Chart</SelectItem>
                      <SelectItem value="area">Area Chart</SelectItem>
                      <SelectItem value="pie">Pie Chart</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div>
                <Label htmlFor="data-source">Data Source</Label>
                <Select
                  value={selectedWidget.config.dataSource}
                  onValueChange={(value) => updateWidget(selectedWidget.id, {
                    config: { ...selectedWidget.config, dataSource: value }
                  })}
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
                <Label htmlFor="refresh-interval">Refresh Interval (ms)</Label>
                <Input
                  id="refresh-interval"
                  type="number"
                  value={selectedWidget.config.refreshInterval || 30000}
                  onChange={(e) => updateWidget(selectedWidget.id, {
                    config: { ...selectedWidget.config, refreshInterval: parseInt(e.target.value) }
                  })}
                />
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground">
              <IconGrid3x3 className="h-12 w-12 mx-auto mb-2" />
              <p>Select a widget to edit its properties</p>
            </div>
          )}
        </div>
      </div>

      {/* Template Selection Dialog */}
      <Dialog open={showTemplates} onOpenChange={setShowTemplates}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Dashboard Templates</DialogTitle>
            <DialogDescription>
              Choose from pre-built templates to get started quickly
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
                      <div className="h-24 bg-muted rounded-md flex items-center justify-center">
                        <IconLayout className="h-8 w-8 text-muted-foreground" />
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {template.category}
                          </Badge>
                          <span className="text-muted-foreground">
                            {template.layout.widgets.length} widgets
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

      {/* Layout Settings Dialog */}
      <Dialog open={showLayoutSettings} onOpenChange={setShowLayoutSettings}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Dashboard Settings</DialogTitle>
            <DialogDescription>
              Configure dashboard layout and behavior
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="dashboard-name">Dashboard Name</Label>
              <Input
                id="dashboard-name"
                value={currentLayout.name}
                onChange={(e) => setCurrentLayout(prev => ({
                  ...prev,
                  name: e.target.value,
                  updatedAt: new Date().toISOString()
                }))}
              />
            </div>

            <div>
              <Label htmlFor="dashboard-description">Description</Label>
              <Textarea
                id="dashboard-description"
                value={currentLayout.description || ''}
                onChange={(e) => setCurrentLayout(prev => ({
                  ...prev,
                  description: e.target.value,
                  updatedAt: new Date().toISOString()
                }))}
                rows={3}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="auto-refresh">Auto Refresh</Label>
              <Switch
                id="auto-refresh"
                checked={currentLayout.settings.autoRefresh}
                onCheckedChange={(checked) => setCurrentLayout(prev => ({
                  ...prev,
                  settings: { ...prev.settings, autoRefresh: checked },
                  updatedAt: new Date().toISOString()
                }))}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="show-grid">Show Grid</Label>
              <Switch
                id="show-grid"
                checked={currentLayout.settings.showGrid}
                onCheckedChange={(checked) => setCurrentLayout(prev => ({
                  ...prev,
                  settings: { ...prev.settings, showGrid: checked },
                  updatedAt: new Date().toISOString()
                }))}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="snap-to-grid">Snap to Grid</Label>
              <Switch
                id="snap-to-grid"
                checked={currentLayout.settings.snapToGrid}
                onCheckedChange={(checked) => setCurrentLayout(prev => ({
                  ...prev,
                  settings: { ...prev.settings, snapToGrid: checked },
                  updatedAt: new Date().toISOString()
                }))}
              />
            </div>

            <div>
              <Label htmlFor="refresh-interval">Refresh Interval (seconds)</Label>
              <Input
                id="refresh-interval"
                type="number"
                value={currentLayout.settings.refreshInterval / 1000}
                onChange={(e) => setCurrentLayout(prev => ({
                  ...prev,
                  settings: { ...prev.settings, refreshInterval: parseInt(e.target.value) * 1000 },
                  updatedAt: new Date().toISOString()
                }))}
              />
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default CustomDashboardBuilder
