'use client'

import { useAuth } from '@/hooks/useAuth'
import { useTranslations } from 'next-intl'
import { Button } from '@/components/ui/button'
import {
  IconLogout,
  IconUser,
  IconSettings,
  IconChevronDown
} from '@tabler/icons-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { toast } from '@/hooks/use-toast'

interface AuthenticatedLayoutProps {
  children: React.ReactNode
  showUserMenu?: boolean
  className?: string
}

export default function AuthenticatedLayout({
  children,
  showUserMenu = true,
  className = ''
}: AuthenticatedLayoutProps) {
  const { user, logout, isLoading } = useAuth()
  const t = useTranslations('Auth')

  const handleLogout = async () => {
    try {
      await logout()
      toast({
        title: t('logoutSuccess'),
        description: 'You have been successfully signed out',
        variant: 'default',
      })
    } catch (error) {
      console.error('Logout error:', error)
      toast({
        title: 'Logout failed',
        description: 'There was an error signing out. Please try again.',
        variant: 'destructive',
      })
    }
  }

  const UserMenu = () => {
    if (!showUserMenu || !user) return null

    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            className="relative h-9 w-auto px-3 py-2 text-sm font-normal"
            disabled={isLoading}
          >
            <div className="flex items-center space-x-2">
              <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-primary">
                <IconUser className="w-3 h-3" aria-hidden="true" />
              </div>
              <div className="flex flex-col items-start">
                <span className="text-sm font-medium">{user.name}</span>
                <span className="text-xs text-muted-foreground capitalize">{user.role}</span>
              </div>
              <IconChevronDown className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
            </div>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56" align="end" forceMount>
          <DropdownMenuLabel className="font-normal">
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-medium">{user.name}</p>
              <p className="text-xs text-muted-foreground">{user.email}</p>
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="cursor-pointer">
            <IconUser className="mr-2 h-4 w-4" aria-hidden="true" />
            Profile
          </DropdownMenuItem>
          <DropdownMenuItem className="cursor-pointer">
            <IconSettings className="mr-2 h-4 w-4" aria-hidden="true" />
            Settings
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="cursor-pointer text-destructive focus:text-destructive"
            onClick={handleLogout}
            disabled={isLoading}
          >
            <IconLogout className="mr-2 h-4 w-4" aria-hidden="true" />
            {t('logout')}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  return (
    <div className={`authenticated-layout ${className}`}>
      {/* User Menu in Top Right */}
      {showUserMenu && (
        <div className="fixed top-4 right-4 z-50">
          <UserMenu />
        </div>
      )}

      {/* Main Content */}
      <div className="authenticated-content">
        {children}
      </div>
    </div>
  )
}

// Hook for authenticated page components
export function useAuthenticatedUser() {
  const { user, isAuthenticated, isLoading } = useAuth()

  return {
    user: user!,
    isAuthenticated,
    isLoading,
    displayName: user?.name || 'Unknown User',
    email: user?.email || '',
    role: user?.role || 'user',
    permissions: user?.permissions || [],
    isActive: user?.is_active ?? false,
    isVerified: user?.is_verified ?? false
  }
}
