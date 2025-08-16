'use client'

import React, { useState, useCallback, useMemo, useTransition, useEffect } from 'react'
import { FixedSizeGrid as Grid } from 'react-window'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  IconSearch,
  IconFilter,
  IconGrid3x3,
  IconGridDots,
  IconRefresh,
  IconSettings,
  IconCheck,
  IconX,
  IconLoader2,
  IconCamera,
  IconAlertTriangle,
  IconChevronDown
} from '@tabler/icons-react'
import { cameraUtils, Camera, APIError } from '@/lib/api'
import { useCameraEvents } from '@/hooks/useRealTimeData'
import CameraCard from './camera-card'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { useToast } from '@/hooks/use-toast'

interface MultiCameraGridProps {
  onCameraSelect?: (camera: Camera) => void
  onCameraConfigure?: (camera: Camera) => void
  showBulkActions?: boolean
  gridSize?: 'small' | 'medium' | 'large'
  autoRefresh?: boolean
  className?: string
}

interface FilterState {
  search: string
  status: 'all' | 'online' | 'offline' | 'maintenance'
  location: string
  sortBy: 'name' | 'status' | 'location' | 'lastActivity'
  sortOrder: 'asc' | 'desc'
}

interface GridSettings {
  itemsPerRow: number
  itemWidth: number
  itemHeight: number
  gap: number
}

const INITIAL_FILTER_STATE: FilterState = {
  search: '',
  status: 'all',
  location: '',
  sortBy: 'name',
  sortOrder: 'asc'
}

const GRID_CONFIGURATIONS = {
  small: { itemsPerRow: 6, itemWidth: 280, itemHeight: 320, gap: 16 },
  medium: { itemsPerRow: 4, itemWidth: 320, itemHeight: 360, gap: 20 },
  large: { itemsPerRow: 3, itemWidth: 400, itemHeight: 420, gap: 24 }
}

export default function MultiCameraGrid({
  onCameraSelect,
  onCameraConfigure,
  showBulkActions = true,
  gridSize = 'medium',
  autoRefresh = true,
  className = ""
}: MultiCameraGridProps) {
  const [isPending, startTransition] = useTransition()
  const [filters, setFilters] = useState<FilterState>(INITIAL_FILTER_STATE)
  const [selectedCameras, setSelectedCameras] = useState<Set<string>>(new Set())
  const [showFilters, setShowFilters] = useState(false)
  const [containerSize, setContainerSize] = useState({ width: 1200, height: 600 })

  const queryClient = useQueryClient()
  const { toast } = useToast()

  // Grid configuration
  const gridConfig = GRID_CONFIGURATIONS[gridSize]

  // Fetch cameras data
  const {
    data: cameras = [],
    isLoading,
    isError,
    error: queryError,
    refetch
  } = useQuery({
    queryKey: ['cameras'],
    queryFn: cameraUtils.getAll,
    refetchInterval: autoRefresh ? 30000 : false,
    staleTime: 15000,
    retry: 3
  })

  // Real-time events for all cameras
  const allCameraEvents = useCameraEvents()

  // Filter and sort cameras
  const filteredCameras = useMemo(() => {
    let filtered = [...cameras]

    // Search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase()
      filtered = filtered.filter(camera =>
        camera.name.toLowerCase().includes(searchLower) ||
        camera.location.toLowerCase().includes(searchLower) ||
        camera.id.toLowerCase().includes(searchLower)
      )
    }

    // Status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(camera => camera.status === filters.status)
    }

    // Location filter
    if (filters.location) {
      const locationLower = filters.location.toLowerCase()
      filtered = filtered.filter(camera =>
        camera.location.toLowerCase().includes(locationLower)
      )
    }

    // Sort cameras
    filtered.sort((a, b) => {
      let compareValue = 0

      switch (filters.sortBy) {
        case 'name':
          compareValue = a.name.localeCompare(b.name)
          break
        case 'status':
          compareValue = a.status.localeCompare(b.status)
          break
        case 'location':
          compareValue = a.location.localeCompare(b.location)
          break
        case 'lastActivity':
          // Would need lastActivity timestamp from API
          compareValue = 0
          break
      }

      return filters.sortOrder === 'desc' ? -compareValue : compareValue
    })

    return filtered
  }, [cameras, filters])

  // Calculate grid dimensions
  const { rowCount, columnCount } = useMemo(() => {
    const itemsPerRow = Math.floor((containerSize.width - gridConfig.gap) / (gridConfig.itemWidth + gridConfig.gap))
    const actualItemsPerRow = Math.max(1, Math.min(itemsPerRow, gridConfig.itemsPerRow))
    const rowCount = Math.ceil(filteredCameras.length / actualItemsPerRow)

    return {
      rowCount,
      columnCount: actualItemsPerRow
    }
  }, [filteredCameras.length, containerSize.width, gridConfig])

  // Grid item renderer
  const GridItem = useCallback(({ columnIndex, rowIndex, style }: any) => {
    const itemIndex = rowIndex * columnCount + columnIndex
    const camera = filteredCameras[itemIndex]

    if (!camera) {
      return <div style={style} />
    }

    const isSelected = selectedCameras.has(camera.id)

    return (
      <div style={{ ...style, padding: gridConfig.gap / 2 }}>
        <div className="relative">
          {showBulkActions && (
            <Checkbox
              checked={isSelected}
              onCheckedChange={(checked) => {
                const newSelected = new Set(selectedCameras)
                if (checked) {
                  newSelected.add(camera.id)
                } else {
                  newSelected.delete(camera.id)
                }
                setSelectedCameras(newSelected)
              }}
              className="absolute top-2 left-2 z-10 bg-white/90"
            />
          )}
          <CameraCard
            cameraId={camera.id}
            onView={() => onCameraSelect?.(camera)}
            onConfigure={() => onCameraConfigure?.(camera)}
            className={`transition-all duration-200 ${
              isSelected ? 'ring-2 ring-primary shadow-lg' : ''
            }`}
          />
        </div>
      </div>
    )
  }, [
    filteredCameras,
    columnCount,
    selectedCameras,
    showBulkActions,
    gridConfig.gap,
    onCameraSelect,
    onCameraConfigure
  ])

  // Handle bulk actions
  const handleBulkAction = useCallback(async (action: string) => {
    if (selectedCameras.size === 0) return

    const cameraIds = Array.from(selectedCameras)

    startTransition(async () => {
      try {
        switch (action) {
          case 'start':
            await Promise.all(cameraIds.map(id => cameraUtils.startStream?.(id)))
            toast({
              title: "Streams Started",
              description: `Started ${cameraIds.length} camera streams.`
            })
            break
          case 'stop':
            await Promise.all(cameraIds.map(id => cameraUtils.stopStream?.(id)))
            toast({
              title: "Streams Stopped",
              description: `Stopped ${cameraIds.length} camera streams.`
            })
            break
          case 'refresh':
            cameraIds.forEach(id => {
              queryClient.invalidateQueries({ queryKey: ['camera', id] })
            })
            toast({
              title: "Cameras Refreshed",
              description: `Refreshed ${cameraIds.length} cameras.`
            })
            break
        }

        // Clear selection after action
        setSelectedCameras(new Set())

        // Refresh main query
        queryClient.invalidateQueries({ queryKey: ['cameras'] })

      } catch (error) {
        const errorMessage = error instanceof APIError
          ? error.message
          : `Failed to ${action} cameras`
        toast({
          title: "Bulk Action Failed",
          description: errorMessage,
          variant: "destructive"
        })
      }
    })
  }, [selectedCameras, queryClient, toast])

  // Handle container resize
  useEffect(() => {
    const handleResize = () => {
      if (typeof window !== 'undefined') {
        const container = document.getElementById('camera-grid-container')
        if (container) {
          setContainerSize({
            width: container.offsetWidth,
            height: container.offsetHeight
          })
        }
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Toggle all cameras selection
  const toggleAllSelection = useCallback(() => {
    if (selectedCameras.size === filteredCameras.length) {
      setSelectedCameras(new Set())
    } else {
      setSelectedCameras(new Set(filteredCameras.map(camera => camera.id)))
    }
  }, [selectedCameras.size, filteredCameras])

  // Get unique locations for filter
  const uniqueLocations = useMemo(() => {
    const locations = cameras.map(camera => camera.location)
    return [...new Set(locations)].sort()
  }, [cameras])

  const isLoading = isLoading || isPending

  if (isError) {
    const errorMessage = queryError instanceof APIError
      ? queryError.message
      : 'Failed to load cameras'

    return (
      <Card className={`p-8 text-center ${className}`}>
        <IconAlertTriangle className="w-12 h-12 text-destructive mx-auto mb-4" />
        <h3 className="text-lg font-medium mb-2">Failed to Load Cameras</h3>
        <p className="text-muted-foreground mb-4">{errorMessage}</p>
        <Button onClick={() => refetch()} variant="outline">
          <IconRefresh className="w-4 h-4 mr-2" />
          Retry
        </Button>
      </Card>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header with Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="text-xl font-semibold">
            Camera Grid
            <Badge variant="secondary" className="ml-2">
              {filteredCameras.length} cameras
            </Badge>
          </h2>
          {isLoading && <IconLoader2 className="w-4 h-4 animate-spin" />}
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
          >
            <IconFilter className="w-4 h-4 mr-2" />
            Filters
            <IconChevronDown className={`w-4 h-4 ml-2 transition-transform ${
              showFilters ? 'rotate-180' : ''
            }`} />
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <IconRefresh className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <Card className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Search</label>
              <Input
                placeholder="Search cameras..."
                value={filters.search}
                onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Status</label>
              <Select
                value={filters.status}
                onValueChange={(value: any) => setFilters(prev => ({ ...prev, status: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="online">Online</SelectItem>
                  <SelectItem value="offline">Offline</SelectItem>
                  <SelectItem value="maintenance">Maintenance</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Location</label>
              <Select
                value={filters.location}
                onValueChange={(value) => setFilters(prev => ({ ...prev, location: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All locations" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Locations</SelectItem>
                  {uniqueLocations.map(location => (
                    <SelectItem key={location} value={location}>
                      {location}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Sort By</label>
              <div className="flex space-x-2">
                <Select
                  value={filters.sortBy}
                  onValueChange={(value: any) => setFilters(prev => ({ ...prev, sortBy: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name">Name</SelectItem>
                    <SelectItem value="status">Status</SelectItem>
                    <SelectItem value="location">Location</SelectItem>
                    <SelectItem value="lastActivity">Last Activity</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setFilters(prev => ({
                    ...prev,
                    sortOrder: prev.sortOrder === 'asc' ? 'desc' : 'asc'
                  }))}
                >
                  {filters.sortOrder === 'asc' ? '↑' : '↓'}
                </Button>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Bulk Actions */}
      {showBulkActions && selectedCameras.size > 0 && (
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Checkbox
                checked={selectedCameras.size === filteredCameras.length}
                indeterminate={selectedCameras.size > 0 && selectedCameras.size < filteredCameras.length}
                onCheckedChange={toggleAllSelection}
              />
              <span className="text-sm font-medium">
                {selectedCameras.size} camera{selectedCameras.size !== 1 ? 's' : ''} selected
              </span>
            </div>

            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleBulkAction('start')}
                disabled={isLoading}
              >
                <IconCamera className="w-4 h-4 mr-2" />
                Start Streams
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleBulkAction('stop')}
                disabled={isLoading}
              >
                <IconX className="w-4 h-4 mr-2" />
                Stop Streams
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleBulkAction('refresh')}
                disabled={isLoading}
              >
                <IconRefresh className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Virtual Grid */}
      <Card className="overflow-hidden">
        <div
          id="camera-grid-container"
          className="w-full"
          style={{ height: '600px' }}
        >
          {filteredCameras.length > 0 ? (
            <Grid
              columnCount={columnCount}
              columnWidth={gridConfig.itemWidth + gridConfig.gap}
              height={600}
              rowCount={rowCount}
              rowHeight={gridConfig.itemHeight + gridConfig.gap}
              width={containerSize.width}
              itemData={{ cameras: filteredCameras }}
            >
              {GridItem}
            </Grid>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center space-y-4">
                <IconCamera className="w-12 h-12 text-muted-foreground mx-auto" />
                <div>
                  <h3 className="font-medium mb-1">No Cameras Found</h3>
                  <p className="text-sm text-muted-foreground">
                    {cameras.length === 0
                      ? 'No cameras are configured yet'
                      : 'No cameras match your current filters'
                    }
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Grid Size Controls */}
      <div className="flex justify-end">
        <Select value={gridSize} onValueChange={(value: any) => setGridSize?.(value)}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="small">Small Grid</SelectItem>
            <SelectItem value="medium">Medium Grid</SelectItem>
            <SelectItem value="large">Large Grid</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
