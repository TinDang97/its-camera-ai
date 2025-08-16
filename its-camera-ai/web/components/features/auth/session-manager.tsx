'use client'

import React, { useEffect, useCallback, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  IconClock,
  IconShieldX,
  IconRefresh,
  IconLogout,
  IconAlertTriangle
} from '@tabler/icons-react'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Alert } from '@/components/ui/alert'

interface SessionManagerProps {
  warningTimeMinutes?: number
  autoRefreshEnabled?: boolean
  showSessionWarnings?: boolean
  onSessionExpired?: () => void
  onSessionExtended?: () => void
}

interface SessionState {
  timeUntilExpiry: number | null
  showExpiryWarning: boolean
  showExpiredDialog: boolean
  isRefreshing: boolean
  refreshError: string | null
  lastActivity: number
}

const DEFAULT_WARNING_TIME = 5 * 60 * 1000 // 5 minutes in milliseconds
const REFRESH_BUFFER_TIME = 30 * 1000 // 30 seconds before expiry
const ACTIVITY_TIMEOUT = 30 * 60 * 1000 // 30 minutes of inactivity
const CHECK_INTERVAL = 30 * 1000 // Check every 30 seconds

export default function SessionManager({
  warningTimeMinutes = 5,
  autoRefreshEnabled = true,
  showSessionWarnings = true,
  onSessionExpired,
  onSessionExtended
}: SessionManagerProps) {
  const [sessionState, setSessionState] = useState<SessionState>({
    timeUntilExpiry: null,
    showExpiryWarning: false,
    showExpiredDialog: false,
    isRefreshing: false,
    refreshError: null,
    lastActivity: Date.now()
  })

  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const warningRef = useRef<NodeJS.Timeout | null>(null)
  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const { isAuthenticated, refreshUser, logout } = useAuth()
  const router = useRouter()

  const warningTime = warningTimeMinutes * 60 * 1000

  // Get token expiry time from localStorage
  const getTokenExpiryTime = useCallback((): number | null => {
    if (typeof window === 'undefined') return null

    try {
      const tokenString = localStorage.getItem('auth_tokens')
      if (!tokenString) return null

      const tokens = JSON.parse(tokenString)
      if (!tokens.access_token) return null

      // Decode JWT payload
      const payload = JSON.parse(atob(tokens.access_token.split('.')[1]))
      return payload.exp * 1000 // Convert to milliseconds
    } catch {
      return null
    }
  }, [])

  // Update last activity timestamp
  const updateLastActivity = useCallback(() => {
    setSessionState(prev => ({
      ...prev,
      lastActivity: Date.now()
    }))
  }, [])

  // Check if user has been inactive
  const isUserInactive = useCallback(() => {
    return Date.now() - sessionState.lastActivity > ACTIVITY_TIMEOUT
  }, [sessionState.lastActivity])

  // Format time remaining for display
  const formatTimeRemaining = useCallback((milliseconds: number): string => {
    const minutes = Math.floor(milliseconds / (1000 * 60))
    const seconds = Math.floor((milliseconds % (1000 * 60)) / 1000)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }, [])

  // Handle session expiry
  const handleSessionExpired = useCallback(async () => {
    setSessionState(prev => ({
      ...prev,
      showExpiryWarning: false,
      showExpiredDialog: true
    }))

    // Clear all timers
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (warningRef.current) clearTimeout(warningRef.current)
    if (refreshTimeoutRef.current) clearTimeout(refreshTimeoutRef.current)

    onSessionExpired?.()

    // Auto-logout after showing expired dialog
    setTimeout(() => {
      logout()
      router.replace('/login?reason=session_expired')
    }, 3000)
  }, [onSessionExpired, logout, router])

  // Refresh session token
  const handleRefreshSession = useCallback(async () => {
    if (sessionState.isRefreshing) return

    setSessionState(prev => ({
      ...prev,
      isRefreshing: true,
      refreshError: null
    }))

    try {
      await refreshUser()

      setSessionState(prev => ({
        ...prev,
        isRefreshing: false,
        showExpiryWarning: false,
        refreshError: null,
        lastActivity: Date.now()
      }))

      onSessionExtended?.()
    } catch (error) {
      console.error('Session refresh failed:', error)

      setSessionState(prev => ({
        ...prev,
        isRefreshing: false,
        refreshError: error instanceof Error ? error.message : 'Failed to refresh session'
      }))

      // If refresh fails, treat as expired
      handleSessionExpired()
    }
  }, [sessionState.isRefreshing, refreshUser, onSessionExtended, handleSessionExpired])

  // Check session status
  const checkSessionStatus = useCallback(() => {
    if (!isAuthenticated) return

    const expiryTime = getTokenExpiryTime()
    if (!expiryTime) return

    const now = Date.now()
    const timeUntilExpiry = expiryTime - now

    setSessionState(prev => ({
      ...prev,
      timeUntilExpiry
    }))

    // If token has already expired
    if (timeUntilExpiry <= 0) {
      handleSessionExpired()
      return
    }

    // If within warning time and warnings are enabled
    if (timeUntilExpiry <= warningTime && showSessionWarnings) {
      setSessionState(prev => ({
        ...prev,
        showExpiryWarning: true
      }))
    }

    // Auto-refresh if enabled and within refresh buffer time
    if (autoRefreshEnabled && timeUntilExpiry <= REFRESH_BUFFER_TIME && !sessionState.isRefreshing) {
      // Only auto-refresh if user has been active recently
      if (!isUserInactive()) {
        handleRefreshSession()
      } else {
        // User inactive, show expiry warning instead
        setSessionState(prev => ({
          ...prev,
          showExpiryWarning: true
        }))
      }
    }
  }, [
    isAuthenticated,
    getTokenExpiryTime,
    warningTime,
    showSessionWarnings,
    autoRefreshEnabled,
    sessionState.isRefreshing,
    isUserInactive,
    handleSessionExpired,
    handleRefreshSession
  ])

  // Set up session monitoring
  useEffect(() => {
    if (!isAuthenticated) {
      // Clear all timers when not authenticated
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (warningRef.current) clearTimeout(warningRef.current)
      if (refreshTimeoutRef.current) clearTimeout(refreshTimeoutRef.current)
      return
    }

    // Initial check
    checkSessionStatus()

    // Set up interval to check session status
    intervalRef.current = setInterval(checkSessionStatus, CHECK_INTERVAL)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isAuthenticated, checkSessionStatus])

  // Track user activity
  useEffect(() => {
    const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click']

    const handleActivity = () => updateLastActivity()

    // Add event listeners for user activity
    activityEvents.forEach(event => {
      document.addEventListener(event, handleActivity, true)
    })

    return () => {
      activityEvents.forEach(event => {
        document.removeEventListener(event, handleActivity, true)
      })
    }
  }, [updateLastActivity])

  // Handle storage events for cross-tab synchronization
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'auth_tokens') {
        if (e.newValue === null) {
          // Tokens were cleared in another tab
          setSessionState(prev => ({
            ...prev,
            showExpiryWarning: false,
            showExpiredDialog: false
          }))
        } else {
          // Tokens were updated in another tab, refresh our session check
          checkSessionStatus()
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [checkSessionStatus])

  // Don't render anything if not authenticated
  if (!isAuthenticated) {
    return null
  }

  return (
    <>
      {/* Session Expiry Warning Dialog */}
      <Dialog open={sessionState.showExpiryWarning} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-md" hideCloseButton>
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <IconClock className="h-5 w-5 text-orange-500" />
              <span>Session Expiring Soon</span>
            </DialogTitle>
            <DialogDescription className="space-y-2">
              <p>
                Your session will expire in{' '}
                <span className="font-mono font-bold text-foreground">
                  {sessionState.timeUntilExpiry
                    ? formatTimeRemaining(sessionState.timeUntilExpiry)
                    : '0:00'
                  }
                </span>
              </p>
              <p>Would you like to extend your session?</p>
            </DialogDescription>
          </DialogHeader>

          {sessionState.refreshError && (
            <Alert variant="destructive" className="mt-4">
              <IconAlertTriangle className="h-4 w-4" />
              <p className="text-sm">{sessionState.refreshError}</p>
            </Alert>
          )}

          <DialogFooter className="sm:flex-row sm:justify-end sm:space-x-2">
            <Button
              variant="outline"
              onClick={() => {
                logout()
                router.replace('/login')
              }}
              disabled={sessionState.isRefreshing}
            >
              <IconLogout className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
            <Button
              onClick={handleRefreshSession}
              disabled={sessionState.isRefreshing}
              className="sm:ml-0"
            >
              {sessionState.isRefreshing ? (
                <>
                  <IconRefresh className="w-4 h-4 mr-2 animate-spin" />
                  Extending...
                </>
              ) : (
                <>
                  <IconRefresh className="w-4 h-4 mr-2" />
                  Extend Session
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Session Expired Dialog */}
      <Dialog open={sessionState.showExpiredDialog} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-md" hideCloseButton>
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <IconShieldX className="h-5 w-5 text-destructive" />
              <span>Session Expired</span>
            </DialogTitle>
            <DialogDescription>
              Your session has expired for security reasons. You will be redirected to the login page.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              onClick={() => {
                logout()
                router.replace('/login?reason=session_expired')
              }}
              className="w-full"
            >
              <IconLogout className="w-4 h-4 mr-2" />
              Return to Login
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
