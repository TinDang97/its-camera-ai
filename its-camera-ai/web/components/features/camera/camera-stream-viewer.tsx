'use client'

import React, { useRef, useState, useCallback, useEffect, useTransition } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  IconPlayerPlay,
  IconPlayerPause,
  IconVolume,
  IconVolumeOff,
  IconMaximize,
  IconMinimize,
  IconCamera,
  IconSettings,
  IconRefresh,
  IconDownload,
  IconAlertTriangle,
  IconLoader2,
  IconScreenShare,
  IconScreenShareOff,
  IconRecord,
  IconRecordOff
} from '@tabler/icons-react'
import { cameraUtils, Camera, APIError } from '@/lib/api'
import { useCameraEvents } from '@/hooks/useRealTimeData'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'

interface CameraStreamViewerProps {
  cameraId: string
  streamUrl?: string
  autoPlay?: boolean
  muted?: boolean
  showControls?: boolean
  showStats?: boolean
  fullscreenEnabled?: boolean
  recordingEnabled?: boolean
  onSnapshot?: (imageData: string) => void
  onError?: (error: Error) => void
  className?: string
}

interface StreamState {
  isPlaying: boolean
  isMuted: boolean
  isFullscreen: boolean
  isRecording: boolean
  volume: number
  quality: string
  error: string | null
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'failed'
}

interface StreamStats {
  fps: number
  bitrate: string
  resolution: string
  latency: string
  packetsLost: number
  bufferHealth: number
}

const INITIAL_STREAM_STATE: StreamState = {
  isPlaying: false,
  isMuted: true,
  isFullscreen: false,
  isRecording: false,
  volume: 0.5,
  quality: 'auto',
  error: null,
  connectionState: 'disconnected'
}

const INITIAL_STREAM_STATS: StreamStats = {
  fps: 0,
  bitrate: '0 Kbps',
  resolution: '0x0',
  latency: '0ms',
  packetsLost: 0,
  bufferHealth: 0
}

const QUALITY_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: '1080p', label: '1080p' },
  { value: '720p', label: '720p' },
  { value: '480p', label: '480p' },
  { value: '360p', label: '360p' }
]

export default function CameraStreamViewer({
  cameraId,
  streamUrl,
  autoPlay = false,
  muted = true,
  showControls = true,
  showStats = false,
  fullscreenEnabled = true,
  recordingEnabled = false,
  onSnapshot,
  onError,
  className = ""
}: CameraStreamViewerProps) {
  const [isPending, startTransition] = useTransition()
  const [streamState, setStreamState] = useState<StreamState>({
    ...INITIAL_STREAM_STATE,
    isMuted: muted
  })
  const [streamStats, setStreamStats] = useState<StreamStats>(INITIAL_STREAM_STATS)

  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])

  const { toast } = useToast()

  // Fetch camera data
  const { data: camera } = useQuery({
    queryKey: ['camera', cameraId],
    queryFn: () => cameraUtils.getById(cameraId),
    staleTime: 30000
  })

  // Real-time events
  const cameraEvents = useCameraEvents(cameraId)

  // Start stream mutation
  const startStreamMutation = useMutation({
    mutationFn: async () => {
      if (!camera?.stream_url && !streamUrl) {
        throw new Error('No stream URL available')
      }

      // Start the camera stream on the backend
      await cameraUtils.startStream?.(cameraId)
      return camera?.stream_url || streamUrl!
    },
    onSuccess: (url) => {
      setStreamState(prev => ({
        ...prev,
        connectionState: 'connecting',
        error: null
      }))
      setupVideoStream(url)
    },
    onError: (error) => {
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Failed to start stream'
      setStreamState(prev => ({
        ...prev,
        connectionState: 'failed',
        error: errorMessage
      }))
      onError?.(error instanceof Error ? error : new Error(errorMessage))
    }
  })

  // Setup video stream (WebRTC/HLS)
  const setupVideoStream = useCallback(async (url: string) => {
    if (!videoRef.current) return

    try {
      setStreamState(prev => ({ ...prev, connectionState: 'connecting' }))

      // Check if WebRTC is supported and URL is WebRTC
      if (url.includes('webrtc') && 'RTCPeerConnection' in window) {
        await setupWebRTCStream(url)
      } else if (url.includes('.m3u8')) {
        // HLS Stream
        await setupHLSStream(url)
      } else {
        // Regular video stream
        videoRef.current.src = url
        videoRef.current.load()
      }
    } catch (error) {
      console.error('Failed to setup video stream:', error)
      setStreamState(prev => ({
        ...prev,
        connectionState: 'failed',
        error: 'Failed to load video stream'
      }))
    }
  }, [])

  // Setup WebRTC stream
  const setupWebRTCStream = useCallback(async (url: string) => {
    try {
      const peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      })

      peerConnection.ontrack = (event) => {
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0]
          setStreamState(prev => ({ ...prev, connectionState: 'connected' }))
        }
      }

      peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState
        if (state === 'connected') {
          setStreamState(prev => ({ ...prev, connectionState: 'connected' }))
        } else if (state === 'failed' || state === 'disconnected') {
          setStreamState(prev => ({ ...prev, connectionState: 'failed' }))
        }
      }

      // Create offer and set up WebRTC connection
      const offer = await peerConnection.createOffer()
      await peerConnection.setLocalDescription(offer)

      // Send offer to server and handle answer
      // This would need to be implemented based on your WebRTC signaling server

    } catch (error) {
      console.error('WebRTC setup failed:', error)
      throw error
    }
  }, [])

  // Setup HLS stream
  const setupHLSStream = useCallback(async (url: string) => {
    if (!videoRef.current) return

    try {
      // Check if HLS.js is available
      if (window.Hls && window.Hls.isSupported()) {
        const hls = new window.Hls()
        hls.loadSource(url)
        hls.attachMedia(videoRef.current)

        hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
          setStreamState(prev => ({ ...prev, connectionState: 'connected' }))
        })

        hls.on(window.Hls.Events.ERROR, (event: any, data: any) => {
          console.error('HLS error:', data)
          setStreamState(prev => ({
            ...prev,
            connectionState: 'failed',
            error: 'HLS stream error'
          }))
        })
      } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
        // Native HLS support (Safari)
        videoRef.current.src = url
        videoRef.current.load()
      } else {
        throw new Error('HLS not supported')
      }
    } catch (error) {
      console.error('HLS setup failed:', error)
      throw error
    }
  }, [])

  // Video event handlers
  const handlePlay = useCallback(() => {
    if (videoRef.current) {
      if (streamState.isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
    }
  }, [streamState.isPlaying])

  const handleVolumeChange = useCallback((value: number[]) => {
    const volume = value[0]
    setStreamState(prev => ({ ...prev, volume }))
    if (videoRef.current) {
      videoRef.current.volume = volume
    }
  }, [])

  const handleMuteToggle = useCallback(() => {
    setStreamState(prev => ({ ...prev, isMuted: !prev.isMuted }))
    if (videoRef.current) {
      videoRef.current.muted = !streamState.isMuted
    }
  }, [streamState.isMuted])

  const handleFullscreen = useCallback(async () => {
    if (!fullscreenEnabled || !containerRef.current) return

    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen()
        setStreamState(prev => ({ ...prev, isFullscreen: true }))
      } else {
        await document.exitFullscreen()
        setStreamState(prev => ({ ...prev, isFullscreen: false }))
      }
    } catch (error) {
      console.error('Fullscreen error:', error)
    }
  }, [fullscreenEnabled])

  const handleSnapshot = useCallback(() => {
    if (!videoRef.current || !onSnapshot) return

    const canvas = document.createElement('canvas')
    const video = videoRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.drawImage(video, 0, 0)
      const imageData = canvas.toDataURL('image/jpeg', 0.9)
      onSnapshot(imageData)

      toast({
        title: "Snapshot Taken",
        description: "Camera snapshot has been captured successfully."
      })
    }
  }, [onSnapshot, toast])

  const handleRecording = useCallback(() => {
    if (!recordingEnabled || !videoRef.current) return

    if (streamState.isRecording) {
      // Stop recording
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop()
      }
    } else {
      // Start recording
      if (videoRef.current.srcObject instanceof MediaStream) {
        const mediaRecorder = new MediaRecorder(videoRef.current.srcObject)
        recordedChunksRef.current = []

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            recordedChunksRef.current.push(event.data)
          }
        }

        mediaRecorder.onstop = () => {
          const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' })
          const url = URL.createObjectURL(blob)

          // Create download link
          const a = document.createElement('a')
          a.href = url
          a.download = `camera-${cameraId}-${Date.now()}.webm`
          a.click()

          URL.revokeObjectURL(url)
          setStreamState(prev => ({ ...prev, isRecording: false }))

          toast({
            title: "Recording Saved",
            description: "Camera recording has been downloaded."
          })
        }

        mediaRecorderRef.current = mediaRecorder
        mediaRecorder.start()
        setStreamState(prev => ({ ...prev, isRecording: true }))
      }
    }
  }, [recordingEnabled, streamState.isRecording, cameraId, toast])

  // Update stream stats
  useEffect(() => {
    if (!videoRef.current || streamState.connectionState !== 'connected') return

    const updateStats = () => {
      const video = videoRef.current!
      const stats = {
        fps: Math.round(25 + Math.random() * 10), // Mock FPS
        bitrate: `${(Math.random() * 2 + 1).toFixed(1)} Mbps`,
        resolution: `${video.videoWidth}x${video.videoHeight}`,
        latency: `${Math.round(40 + Math.random() * 20)}ms`,
        packetsLost: Math.round(Math.random() * 5),
        bufferHealth: Math.round(80 + Math.random() * 20)
      }
      setStreamStats(stats)
    }

    const interval = setInterval(updateStats, 1000)
    return () => clearInterval(interval)
  }, [streamState.connectionState])

  // Video event listeners
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleCanPlay = () => {
      if (autoPlay) {
        video.play().catch(console.error)
      }
    }

    const handlePlay = () => setStreamState(prev => ({ ...prev, isPlaying: true }))
    const handlePause = () => setStreamState(prev => ({ ...prev, isPlaying: false }))
    const handleError = () => {
      setStreamState(prev => ({
        ...prev,
        connectionState: 'failed',
        error: 'Video playback error'
      }))
    }

    video.addEventListener('canplay', handleCanPlay)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('error', handleError)

    return () => {
      video.removeEventListener('canplay', handleCanPlay)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('error', handleError)
    }
  }, [autoPlay])

  // Auto-start stream
  useEffect(() => {
    if (autoPlay && camera?.stream_url && streamState.connectionState === 'disconnected') {
      startStreamMutation.mutate()
    }
  }, [autoPlay, camera, streamState.connectionState, startStreamMutation])

  const isLoading = startStreamMutation.isPending || isPending
  const hasError = streamState.error || streamState.connectionState === 'failed'

  return (
    <Card ref={containerRef} className={`relative overflow-hidden ${className}`}>
      {/* Video Element */}
      <div className="relative aspect-video bg-black">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          muted={streamState.isMuted}
          volume={streamState.volume}
          playsInline
          controls={false}
        />

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="text-center space-y-2">
              <IconLoader2 className="w-8 h-8 text-white animate-spin mx-auto" />
              <p className="text-white text-sm">Connecting to stream...</p>
            </div>
          </div>
        )}

        {/* Error Overlay */}
        {hasError && (
          <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
            <div className="text-center space-y-4 p-6">
              <IconAlertTriangle className="w-12 h-12 text-destructive mx-auto" />
              <div className="space-y-2">
                <h3 className="text-white font-medium">Stream Unavailable</h3>
                <p className="text-gray-300 text-sm">{streamState.error}</p>
              </div>
              <Button
                variant="secondary"
                onClick={() => startStreamMutation.mutate()}
                disabled={isLoading}
              >
                <IconRefresh className="w-4 h-4 mr-2" />
                Retry Connection
              </Button>
            </div>
          </div>
        )}

        {/* Status Badges */}
        <div className="absolute top-4 left-4 flex space-x-2">
          <Badge
            variant={streamState.connectionState === 'connected' ? 'destructive' : 'secondary'}
            className="text-xs"
          >
            {streamState.connectionState === 'connected' ? (
              <>
                <IconCamera className="w-3 h-3 mr-1" />
                LIVE
              </>
            ) : (
              <>
                <IconAlertTriangle className="w-3 h-3 mr-1" />
                {streamState.connectionState.toUpperCase()}
              </>
            )}
          </Badge>

          {streamState.isRecording && (
            <Badge variant="destructive" className="text-xs animate-pulse">
              <IconRecord className="w-3 h-3 mr-1" />
              REC
            </Badge>
          )}
        </div>

        {/* Stream Stats */}
        {showStats && streamState.connectionState === 'connected' && (
          <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-sm rounded p-2 text-xs text-white space-y-1">
            <div>FPS: {streamStats.fps}</div>
            <div>Bitrate: {streamStats.bitrate}</div>
            <div>Latency: {streamStats.latency}</div>
            <div>Resolution: {streamStats.resolution}</div>
          </div>
        )}

        {/* Controls Overlay */}
        {showControls && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
            <div className="flex items-center space-x-4">
              {/* Play/Pause */}
              <Button
                variant="ghost"
                size="sm"
                onClick={handlePlay}
                disabled={streamState.connectionState !== 'connected'}
                className="text-white hover:bg-white/20"
              >
                {streamState.isPlaying ? (
                  <IconPlayerPause className="w-5 h-5" />
                ) : (
                  <IconPlayerPlay className="w-5 h-5" />
                )}
              </Button>

              {/* Volume Control */}
              <div className="flex items-center space-x-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleMuteToggle}
                  className="text-white hover:bg-white/20"
                >
                  {streamState.isMuted ? (
                    <IconVolumeOff className="w-4 h-4" />
                  ) : (
                    <IconVolume className="w-4 h-4" />
                  )}
                </Button>
                <Slider
                  value={[streamState.volume]}
                  onValueChange={handleVolumeChange}
                  max={1}
                  step={0.1}
                  className="w-20"
                />
              </div>

              {/* Quality Selector */}
              <Select value={streamState.quality} onValueChange={(value) =>
                setStreamState(prev => ({ ...prev, quality: value }))
              }>
                <SelectTrigger className="w-20 h-8 bg-white/10 border-white/20 text-white text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {QUALITY_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <div className="flex-1" />

              {/* Action Buttons */}
              <div className="flex space-x-2">
                {onSnapshot && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleSnapshot}
                    disabled={streamState.connectionState !== 'connected'}
                    className="text-white hover:bg-white/20"
                  >
                    <IconCamera className="w-4 h-4" />
                  </Button>
                )}

                {recordingEnabled && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRecording}
                    disabled={streamState.connectionState !== 'connected'}
                    className="text-white hover:bg-white/20"
                  >
                    {streamState.isRecording ? (
                      <IconRecordOff className="w-4 h-4" />
                    ) : (
                      <IconRecord className="w-4 h-4" />
                    )}
                  </Button>
                )}

                {fullscreenEnabled && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleFullscreen}
                    className="text-white hover:bg-white/20"
                  >
                    {streamState.isFullscreen ? (
                      <IconMinimize className="w-4 h-4" />
                    ) : (
                      <IconMaximize className="w-4 h-4" />
                    )}
                  </Button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Connection Status */}
      <div className={`absolute top-2 right-2 w-3 h-3 rounded-full ${
        streamState.connectionState === 'connected' ? 'bg-online' :
        streamState.connectionState === 'connecting' ? 'bg-secondary animate-pulse' : 'bg-offline'
      }`} />
    </Card>
  )
}

// Add HLS.js type declaration
declare global {
  interface Window {
    Hls: any
  }
}
