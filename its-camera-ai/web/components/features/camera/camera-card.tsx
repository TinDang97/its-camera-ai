'use client'

import React, { useState, useOptimistic, useTransition, useCallback, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Image from 'next/image'
import {
  IconCamera,
  IconPlayerPlay,
  IconPlayerStop,
  IconSettings,
  IconRefresh,
  IconMaximize,
  IconWifi,
  IconWifiOff,
  IconTool,
  IconAlertTriangle,
  IconActivity,
  IconClock,
  IconLoader2,
  IconEye
} from '@tabler/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { cameraUtils, Camera, APIError } from '@/lib/api'
import { useCameraEvents } from '@/hooks/useRealTimeData'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'

interface CameraCardProps {
  cameraId: string
  showActions?: boolean
  showStats?: boolean
  onView?: (camera: Camera) => void
  onConfigure?: (camera: Camera) => void
  className?: string
}

interface CameraState {
  isStreaming: boolean
  error: string | null
  lastActivity: string | null
}

const INITIAL_CAMERA_STATE: CameraState = {
  isStreaming: false,
  error: null,
  lastActivity: null
}

export default function CameraCard({
  cameraId,
  showActions = true,
  showStats = true,
  onView,
  onConfigure,
  className = ""
}: CameraCardProps) {
  const [isPending, startTransition] = useTransition()
  const [cameraState, setCameraState] = useState<CameraState>(INITIAL_CAMERA_STATE)
  const [optimisticState, updateOptimisticState] = useOptimistic(cameraState)
  const [thumbnailError, setThumbnailError] = useState(false)

  const router = useRouter()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  // Fetch camera data
  const {
    data: camera,
    isLoading,
    isError,
    error: queryError
  } = useQuery({
    queryKey: ['camera', cameraId],
    queryFn: () => cameraUtils.getById(cameraId),
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  // Real-time camera events
  const cameraEvents = useCameraEvents(cameraId)

  // Start stream mutation
  const startStreamMutation = useMutation({
    mutationFn: () => cameraUtils.startStream?.(cameraId) || Promise.resolve(),
    onSuccess: () => {
      setCameraState(prev => ({ ...prev, isStreaming: true, error: null }))
      toast({
        title: "Stream Started",
        description: `Camera ${camera?.name || cameraId} stream has been started.`
      })
      // Invalidate camera data to get updated status
      queryClient.invalidateQueries({ queryKey: ['camera', cameraId] })
    },
    onError: (error) => {
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Failed to start camera stream'
      setCameraState(prev => ({ ...prev, error: errorMessage }))
      toast({
        title: "Stream Failed",
        description: errorMessage,
        variant: "destructive"
      })
    }
  })

  // Stop stream mutation
  const stopStreamMutation = useMutation({
    mutationFn: () => cameraUtils.stopStream?.(cameraId) || Promise.resolve(),
    onSuccess: () => {
      setCameraState(prev => ({ ...prev, isStreaming: false, error: null }))
      toast({
        title: "Stream Stopped",
        description: `Camera ${camera?.name || cameraId} stream has been stopped.`
      })
      queryClient.invalidateQueries({ queryKey: ['camera', cameraId] })
    },
    onError: (error) => {
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Failed to stop camera stream'
      setCameraState(prev => ({ ...prev, error: errorMessage }))
      toast({
        title: "Stop Stream Failed",
        description: errorMessage,
        variant: "destructive"
      })
    }
  })

  // Handle real-time events
  useEffect(() => {
    if (cameraEvents.events.length > 0) {
      const latestEvent = cameraEvents.events[0]

      switch (latestEvent.type) {
        case 'status_change':
          // Invalidate query to refetch camera data
          queryClient.invalidateQueries({ queryKey: ['camera', cameraId] })
          setCameraState(prev => ({
            ...prev,
            lastActivity: new Date().toISOString()
          }))
          break

        case 'detection_result':
          setCameraState(prev => ({
            ...prev,
            lastActivity: new Date().toISOString()
          }))
          break

        case 'health_update':
          queryClient.invalidateQueries({ queryKey: ['camera', cameraId] })
          break
      }
    }
  }, [cameraEvents.events, queryClient, cameraId])

  // Status badge variant
  const getStatusBadgeVariant = useCallback((status: string) => {
    switch (status) {
      case 'online':
        return 'default' // Green
      case 'offline':
        return 'destructive' // Red
      case 'maintenance':
        return 'secondary' // Yellow/Orange
      default:
        return 'outline'
    }
  }, [])

  // Status icon
  const getStatusIcon = useCallback((status: string) => {
    switch (status) {
      case 'online':
        return <IconWifi className="w-3 h-3" />
      case 'offline':
        return <IconWifiOff className="w-3 h-3" />
      case 'maintenance':
        return <IconTool className="w-3 h-3" />
      default:
        return <IconAlertTriangle className="w-3 h-3" />
    }
  }, [])

  // Handle thumbnail error
  const handleThumbnailError = useCallback(() => {
    setThumbnailError(true)
  }, [])

  // Handle actions
  const handleStartStream = useCallback(() => {
    startTransition(() => {
      updateOptimisticState(prev => ({ ...prev, isStreaming: true }))
      startStreamMutation.mutate()
    })
  }, [startStreamMutation, updateOptimisticState])

  const handleStopStream = useCallback(() => {
    startTransition(() => {
      updateOptimisticState(prev => ({ ...prev, isStreaming: false }))
      stopStreamMutation.mutate()
    })
  }, [stopStreamMutation, updateOptimisticState])

  const handleView = useCallback(() => {
    if (camera) {
      onView?.(camera)
      router.push(`/cameras/${cameraId}/live`)
    }
  }, [camera, onView, router, cameraId])

  const handleConfigure = useCallback(() => {
    if (camera) {
      onConfigure?.(camera)
      router.push(`/cameras/${cameraId}/settings`)
    }
  }, [camera, onConfigure, router, cameraId])

  // Loading state
  if (isLoading) {
    return (
      <Card className={`animate-pulse ${className}`}>
        <div className="aspect-video bg-muted rounded-t-lg" />
        <div className="p-4 space-y-3">
          <div className="h-4 bg-muted rounded w-3/4" />
          <div className="h-3 bg-muted rounded w-1/2" />
          <div className="flex space-x-2">
            <div className="h-6 bg-muted rounded w-16" />
            <div className="h-6 bg-muted rounded w-16" />
          </div>
        </div>
      </Card>
    )
  }

  // Error state
  if (isError || !camera) {
    const errorMessage = queryError instanceof APIError
      ? queryError.message
      : 'Failed to load camera data'

    return (
      <Card className={`border-destructive/50 ${className}`}>
        <div className="aspect-video bg-destructive/10 rounded-t-lg flex items-center justify-center">
          <div className="text-center space-y-2">
            <IconAlertTriangle className="w-8 h-8 text-destructive mx-auto" />
            <p className="text-sm text-destructive font-medium">Camera Unavailable</p>
          </div>
        </div>
        <div className="p-4">
          <h3 className="font-medium text-foreground mb-1">Camera {cameraId}</h3>
          <p className="text-sm text-muted-foreground mb-3">{errorMessage}</p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => queryClient.invalidateQueries({ queryKey: ['camera', cameraId] })}
            className="w-full"
          >
            <IconRefresh className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      </Card>
    )
  }

  const isStreamingActive = optimisticState.isStreaming || camera.status === 'online'
  const isActionLoading = startStreamMutation.isPending || stopStreamMutation.isPending || isPending

  return (
    <Card className={`group hover:shadow-lg transition-all duration-200 ${className}`}>
      {/* Thumbnail/Preview */}
      <div className="relative aspect-video overflow-hidden rounded-t-lg bg-muted">
        {camera.stream_url && !thumbnailError ? (
          <Image
            src={`${camera.stream_url}/thumbnail`}
            alt={`${camera.name} preview`}
            fill
            className="object-cover transition-transform group-hover:scale-105"
            onError={handleThumbnailError}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-muted">
            <IconCamera className="w-12 h-12 text-muted-foreground" />
          </div>
        )}

        {/* Status Overlay */}
        <div className="absolute top-2 left-2">
          <Badge variant={getStatusBadgeVariant(camera.status)} className="text-xs">
            {getStatusIcon(camera.status)}
            <span className="ml-1 capitalize">{camera.status}</span>
          </Badge>
        </div>

        {/* Live Indicator */}
        {isStreamingActive && (
          <div className="absolute top-2 right-2">
            <Badge variant="destructive" className="text-xs animate-pulse">
              <IconActivity className="w-3 h-3 mr-1" />
              LIVE
            </Badge>
          </div>
        )}

        {/* View Button Overlay */}
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleView}
            className="backdrop-blur-sm"
          >
            <IconEye className="w-4 h-4 mr-2" />
            View Live
          </Button>
        </div>
      </div>

      {/* Camera Info */}
      <div className="p-4 space-y-3">
        {/* Header */}
        <div className="space-y-1">
          <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
            {camera.name}
          </h3>
          <p className="text-sm text-muted-foreground">
            {camera.location}
          </p>
        </div>

        {/* Stats */}
        {showStats && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center space-x-1">
              <IconActivity className="w-3 h-3 text-muted-foreground" />
              <span className="text-muted-foreground">FPS:</span>
              <span className="font-medium">{camera.fps}</span>
            </div>
            <div className="flex items-center space-x-1">
              <IconMaximize className="w-3 h-3 text-muted-foreground" />
              <span className="text-muted-foreground">Res:</span>
              <span className="font-medium">
                {camera.resolution.width}x{camera.resolution.height}
              </span>
            </div>
            {cameraState.lastActivity && (
              <div className="flex items-center space-x-1 col-span-2">
                <IconClock className="w-3 h-3 text-muted-foreground" />
                <span className="text-muted-foreground">Last seen:</span>
                <span className="font-medium">
                  {new Date(cameraState.lastActivity).toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {cameraState.error && (
          <div className="p-2 bg-destructive/10 border border-destructive/20 rounded text-xs text-destructive">
            {cameraState.error}
          </div>
        )}

        {/* Actions */}
        {showActions && (
          <div className="flex space-x-2 pt-2 border-t">
            <Button
              variant="outline"
              size="sm"
              onClick={isStreamingActive ? handleStopStream : handleStartStream}
              disabled={isActionLoading}
              className="flex-1"
            >
              {isActionLoading ? (
                <IconLoader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : isStreamingActive ? (
                <IconPlayerStop className="w-4 h-4 mr-2" />
              ) : (
                <IconPlayerPlay className="w-4 h-4 mr-2" />
              )}
              {isActionLoading
                ? 'Processing...'
                : isStreamingActive
                  ? 'Stop'
                  : 'Start'
              }
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleConfigure}
              disabled={isActionLoading}
            >
              <IconSettings className="w-4 h-4" />
            </Button>
          </div>
        )}
      </div>

      {/* Connection Status */}
      <div className={`absolute bottom-2 right-2 w-2 h-2 rounded-full ${
        cameraEvents.isConnected ? 'bg-online' : 'bg-offline'
      }`} />
    </Card>
  )
}
