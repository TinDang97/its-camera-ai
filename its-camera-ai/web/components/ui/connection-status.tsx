'use client'

import { useWebSocket } from '@/components/providers/websocket-provider'
import { Badge } from '@/components/ui/badge'
import { Wifi, WifiOff, RotateCcw, AlertTriangle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ConnectionStatusProps {
  className?: string
  showText?: boolean
}

export function ConnectionStatus({ className, showText = true }: ConnectionStatusProps) {
  const { connectionState, isConnected, reconnectAttempts } = useWebSocket()

  const getStatusConfig = () => {
    switch (connectionState) {
      case 'connected':
        return {
          icon: Wifi,
          text: 'Connected',
          variant: 'default' as const,
          className: 'bg-green-100 text-green-800 border-green-300'
        }
      case 'connecting':
        return {
          icon: RotateCcw,
          text: 'Connecting...',
          variant: 'secondary' as const,
          className: 'bg-yellow-100 text-yellow-800 border-yellow-300'
        }
      case 'disconnected':
        return {
          icon: WifiOff,
          text: reconnectAttempts > 0 ? `Offline (${reconnectAttempts} attempts)` : 'Demo Mode',
          variant: 'outline' as const,
          className: 'bg-blue-100 text-blue-800 border-blue-300'
        }
      case 'error':
        return {
          icon: AlertTriangle,
          text: 'Connection Error',
          variant: 'destructive' as const,
          className: 'bg-red-100 text-red-800 border-red-300'
        }
      default:
        return {
          icon: WifiOff,
          text: 'Unknown',
          variant: 'outline' as const,
          className: 'bg-gray-100 text-gray-800 border-gray-300'
        }
    }
  }

  const config = getStatusConfig()
  const Icon = config.icon

  return (
    <Badge 
      variant={config.variant}
      className={cn(config.className, 'flex items-center gap-1', className)}
    >
      <Icon className="h-3 w-3" />
      {showText && <span>{config.text}</span>}
    </Badge>
  )
}