'use client'

import React, { useState, useCallback, useEffect, useTransition } from 'react'
import { useRouter } from 'next/navigation'
import {
  IconUser,
  IconSettings,
  IconShield,
  IconLogout,
  IconChevronDown,
  IconBell,
  IconActivity,
  IconClock,
  IconUserCheck,
  IconUserX,
  IconLoader2
} from '@tabler/icons-react'
import { useAuth, usePermissions } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator, DropdownMenuLabel } from '@/components/ui/dropdown-menu'
import { Badge } from '@/components/ui/badge'

interface UserProfileDropdownProps {
  showFullName?: boolean
  showRole?: boolean
  className?: string
}

export default function UserProfileDropdown({
  showFullName = true,
  showRole = true,
  className = ""
}: UserProfileDropdownProps) {
  const [isPending, startTransition] = useTransition()
  const [isOpen, setIsOpen] = useState(false)

  const { user, logout, isAuthenticated, isLoading } = useAuth()
  const { hasPermission, hasRole } = usePermissions()
  const router = useRouter()

  // Generate user avatar initials
  const getInitials = useCallback((name: string) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2)
  }, [])

  // Get role badge color
  const getRoleBadgeVariant = useCallback((role: string) => {
    switch (role.toLowerCase()) {
      case 'admin':
      case 'administrator':
        return 'destructive' as const
      case 'operator':
      case 'manager':
        return 'default' as const
      case 'viewer':
      case 'guest':
        return 'secondary' as const
      default:
        return 'outline' as const
    }
  }, [])

  // Handle logout with confirmation
  const handleLogout = useCallback(async () => {
    startTransition(async () => {
      try {
        await logout()
        router.replace('/login')
      } catch (error) {
        console.error('Logout failed:', error)
        // Force logout on client side even if server logout fails
        router.replace('/login')
      }
    })
  }, [logout, router])

  // Handle navigation
  const handleNavigate = useCallback((path: string) => {
    setIsOpen(false)
    router.push(path)
  }, [router])

  // Format last login time
  const formatLastLogin = useCallback((lastLogin?: string) => {
    if (!lastLogin) return 'Never'

    const date = new Date(lastLogin)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))

    if (diffHours < 1) {
      const diffMinutes = Math.floor(diffMs / (1000 * 60))
      return `${diffMinutes} minutes ago`
    } else if (diffHours < 24) {
      return `${diffHours} hours ago`
    } else {
      const diffDays = Math.floor(diffHours / 24)
      return `${diffDays} days ago`
    }
  }, [])

  // Don't render if not authenticated or loading
  if (!isAuthenticated || isLoading || !user) {
    return null
  }

  const canManageSettings = hasPermission('settings:read') || hasRole('admin')
  const canViewAuditLogs = hasPermission('audit:read') || hasRole('admin')
  const canManageNotifications = hasPermission('notifications:manage')

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className={`h-auto p-2 hover:bg-accent/50 focus:bg-accent/50 ${className}`}
          aria-label={`User menu for ${user.name}`}
          aria-expanded={isOpen}
          aria-haspopup="menu"
        >
          <div className="flex items-center space-x-3">
            {/* Avatar */}
            <div className="relative">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                {getInitials(user.name)}
              </div>
              {/* Online Status Indicator */}
              <div
                className="absolute -bottom-1 -right-1 w-3 h-3 bg-online border-2 border-background rounded-full"
                aria-label="Online"
              />
            </div>

            {/* User Info */}
            <div className="flex-1 text-left min-w-0">
              {showFullName && (
                <div className="text-sm font-medium text-foreground truncate">
                  {user.name}
                </div>
              )}
              {showRole && user.role && (
                <div className="text-xs text-muted-foreground">
                  {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                </div>
              )}
            </div>

            <IconChevronDown
              className={`w-4 h-4 transition-transform duration-200 ${
                isOpen ? 'rotate-180' : ''
              }`}
              aria-hidden="true"
            />
          </div>
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent
        className="w-80 p-0"
        align="end"
        sideOffset={8}
      >
        {/* User Header */}
        <div className="p-4 border-b">
          <div className="flex items-start space-x-3">
            <div className="w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-lg font-medium">
              {getInitials(user.name)}
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-semibold text-foreground truncate">
                {user.name}
              </h4>
              <p className="text-xs text-muted-foreground truncate">
                {user.email}
              </p>
              <div className="flex items-center space-x-2 mt-2">
                <Badge variant={getRoleBadgeVariant(user.role)} className="text-xs">
                  {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                </Badge>
                {user.is_verified ? (
                  <Badge variant="outline" className="text-xs text-online">
                    <IconUserCheck className="w-3 h-3 mr-1" />
                    Verified
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-xs text-destructive">
                    <IconUserX className="w-3 h-3 mr-1" />
                    Unverified
                  </Badge>
                )}
              </div>
            </div>
          </div>

          {/* Last Login */}
          {user.last_login && (
            <div className="flex items-center space-x-2 mt-3 text-xs text-muted-foreground">
              <IconClock className="w-3 h-3" />
              <span>Last login: {formatLastLogin(user.last_login)}</span>
            </div>
          )}
        </div>

        {/* Menu Items */}
        <div className="py-2">
          <DropdownMenuItem
            onSelect={() => handleNavigate('/profile')}
            className="px-4 py-2 cursor-pointer"
          >
            <IconUser className="w-4 h-4 mr-3" />
            My Profile
          </DropdownMenuItem>

          {canManageSettings && (
            <DropdownMenuItem
              onSelect={() => handleNavigate('/settings')}
              className="px-4 py-2 cursor-pointer"
            >
              <IconSettings className="w-4 h-4 mr-3" />
              Settings
            </DropdownMenuItem>
          )}

          {canManageNotifications && (
            <DropdownMenuItem
              onSelect={() => handleNavigate('/notifications')}
              className="px-4 py-2 cursor-pointer"
            >
              <IconBell className="w-4 h-4 mr-3" />
              Notifications
            </DropdownMenuItem>
          )}

          <DropdownMenuItem
            onSelect={() => handleNavigate('/security')}
            className="px-4 py-2 cursor-pointer"
          >
            <IconShield className="w-4 h-4 mr-3" />
            Security
          </DropdownMenuItem>

          {canViewAuditLogs && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="px-4 py-2 text-xs font-medium text-muted-foreground">
                Administration
              </DropdownMenuLabel>
              <DropdownMenuItem
                onSelect={() => handleNavigate('/admin/audit')}
                className="px-4 py-2 cursor-pointer"
              >
                <IconActivity className="w-4 h-4 mr-3" />
                Audit Logs
              </DropdownMenuItem>
            </>
          )}
        </div>

        {/* Account Status */}
        <div className="border-t p-4">
          <div className="text-xs text-muted-foreground space-y-1">
            <div className="flex justify-between">
              <span>Account Status:</span>
              <span className={user.is_active ? 'text-online' : 'text-destructive'}>
                {user.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Member Since:</span>
              <span>
                {new Date(user.created_at).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>

        {/* Logout */}
        <div className="border-t p-2">
          <DropdownMenuItem
            onSelect={handleLogout}
            disabled={isPending}
            className="px-4 py-2 cursor-pointer text-destructive hover:text-destructive hover:bg-destructive/10 focus:text-destructive focus:bg-destructive/10"
          >
            {isPending ? (
              <IconLoader2 className="w-4 h-4 mr-3 animate-spin" />
            ) : (
              <IconLogout className="w-4 h-4 mr-3" />
            )}
            {isPending ? 'Signing out...' : 'Sign Out'}
          </DropdownMenuItem>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
