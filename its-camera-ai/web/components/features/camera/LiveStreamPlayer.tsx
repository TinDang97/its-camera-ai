'use client'

import { useRef, useState, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import {
  Play, Pause, Volume2, VolumeX, Maximize, Minimize,
  RotateCcw, Camera, Download, Settings, Activity
} from 'lucide-react'

interface LiveStreamPlayerProps {
  streamUrl: string
  cameraId: string
  cameraName: string
  showControls?: boolean
  autoPlay?: boolean
  muted?: boolean
  onSnapshot?: (imageData: string) => void
}

export function LiveStreamPlayer({
  streamUrl,
  cameraId,
  cameraName,
  showControls = true,
  autoPlay = true,
  muted = true,
  onSnapshot
}: LiveStreamPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(autoPlay)
  const [isMuted, setIsMuted] = useState(muted)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [volume, setVolume] = useState([0.5])
  const [streamStats, setStreamStats] = useState({
    fps: 30,
    bitrate: '2.5 Mbps',
    resolution: '1920x1080',
    latency: '45ms',
    packets: 0
  })

  // Simulate stream statistics updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStreamStats(prev => ({
        ...prev,
        fps: 28 + Math.floor(Math.random() * 4),
        latency: (40 + Math.floor(Math.random() * 20)) + 'ms',
        packets: prev.packets + Math.floor(Math.random() * 100)
      }))
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted
      setIsMuted(!isMuted)
    }
  }

  const handleVolumeChange = (value: number[]) => {
    setVolume(value)
    if (videoRef.current) {
      videoRef.current.volume = value[0]
      if (value[0] === 0) {
        setIsMuted(true)
        videoRef.current.muted = true
      } else if (isMuted) {
        setIsMuted(false)
        videoRef.current.muted = false
      }
    }
  }

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      videoRef.current?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  const handleSnapshot = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0)
        const imageData = canvas.toDataURL('image/png')
        onSnapshot?.(imageData)
        
        // Simulate download
        const link = document.createElement('a')
        link.download = `${cameraId}_${Date.now()}.png`
        link.href = imageData
        link.click()
      }
    }
  }

  const handleReconnect = () => {
    if (videoRef.current) {
      videoRef.current.load()
      if (autoPlay) {
        videoRef.current.play()
      }
    }
  }

  return (
    <Card className="overflow-hidden">
      <div className="relative bg-black">
        {/* Video Element */}
        <div className="aspect-video relative">
          <video
            ref={videoRef}
            className="w-full h-full object-contain"
            autoPlay={autoPlay}
            muted={muted}
            controls={false}
            playsInline
          >
            <source src={streamUrl} type="application/x-mpegURL" />
            {/* Fallback for demo - show placeholder */}
          </video>
          
          {/* Placeholder overlay for demo */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-white/10 text-8xl font-bold">
              LIVE
            </div>
          </div>

          {/* Stream Stats Overlay */}
          <div className="absolute top-2 left-2 space-y-1">
            <Badge variant="outline" className="bg-black/50 text-white border-white/20">
              {cameraName}
            </Badge>
            <div className="flex flex-wrap gap-1">
              <Badge variant="outline" className="bg-black/50 text-white border-white/20 text-xs">
                {streamStats.fps} FPS
              </Badge>
              <Badge variant="outline" className="bg-black/50 text-white border-white/20 text-xs">
                {streamStats.resolution}
              </Badge>
              <Badge variant="outline" className="bg-black/50 text-white border-white/20 text-xs">
                <Activity className="h-3 w-3 mr-1" />
                {streamStats.latency}
              </Badge>
            </div>
          </div>

          {/* Top Right Stats */}
          <div className="absolute top-2 right-2">
            <Badge variant="outline" className="bg-red-600/90 text-white border-red-500">
              • LIVE
            </Badge>
          </div>
        </div>

        {/* Controls */}
        {showControls && (
          <CardContent className="p-3 bg-gray-900">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {/* Play/Pause */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:bg-white/10"
                  onClick={handlePlayPause}
                >
                  {isPlaying ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                </Button>

                {/* Volume */}
                <div className="flex items-center space-x-2">
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-white hover:bg-white/10"
                    onClick={handleMute}
                  >
                    {isMuted || volume[0] === 0 ? (
                      <VolumeX className="h-4 w-4" />
                    ) : (
                      <Volume2 className="h-4 w-4" />
                    )}
                  </Button>
                  <Slider
                    value={volume}
                    onValueChange={handleVolumeChange}
                    max={1}
                    step={0.1}
                    className="w-20"
                  />
                </div>

                {/* Reconnect */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:bg-white/10"
                  onClick={handleReconnect}
                  title="Reconnect"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>

              <div className="flex items-center space-x-2">
                {/* Snapshot */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:bg-white/10"
                  onClick={handleSnapshot}
                  title="Take Snapshot"
                >
                  <Camera className="h-4 w-4" />
                </Button>

                {/* Download */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:bg-white/10"
                  title="Download Recording"
                >
                  <Download className="h-4 w-4" />
                </Button>

                {/* Settings */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:bg-white/10"
                  title="Stream Settings"
                >
                  <Settings className="h-4 w-4" />
                </Button>

                {/* Fullscreen */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:bg-white/10"
                  onClick={handleFullscreen}
                >
                  {isFullscreen ? (
                    <Minimize className="h-4 w-4" />
                  ) : (
                    <Maximize className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {/* Stream Info */}
            <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
              <span>Stream: {streamUrl}</span>
              <span>Bitrate: {streamStats.bitrate} | Packets: {streamStats.packets.toLocaleString()}</span>
            </div>
          </CardContent>
        )}
      </div>
    </Card>
  )
}