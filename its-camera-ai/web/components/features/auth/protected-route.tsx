'use client'

import { useEffect } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth, usePermissions } from '@/hooks/useAuth'
import { IconLoader2, IconLock } from '@tabler/icons-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface ProtectedRouteProps {
  children: React.ReactNode
  requiredPermissions?: string[]
  requiredRole?: string
  fallback?: React.ReactNode
  redirectTo?: string
}

interface AuthLoadingProps {
  message?: string
}

function AuthLoading({ message = 'Checking authentication...' }: AuthLoadingProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <Card className="w-full max-w-md p-8 text-center space-y-4">
        <IconLoader2 className="mx-auto h-8 w-8 animate-spin text-primary" aria-hidden="true" />
        <p className="text-sm text-muted-foreground">{message}</p>
      </Card>
    </div>
  )
}

interface UnauthorizedProps {
  message?: string
  showLoginButton?: boolean
  onLoginRedirect?: () => void
}

function Unauthorized({
  message = 'You do not have permission to access this page.',
  showLoginButton = false,
  onLoginRedirect
}: UnauthorizedProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <Card className="w-full max-w-md p-8 text-center space-y-4">
        <div className="mx-auto w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center">
          <IconLock className="h-8 w-8 text-destructive" aria-hidden="true" />
        </div>
        <div className="space-y-2">
          <h2 className="text-xl font-semibold text-foreground">Access Denied</h2>
          <p className="text-sm text-muted-foreground">{message}</p>
        </div>
        {showLoginButton && (
          <Button onClick={onLoginRedirect} className="w-full">
            Sign in to continue
          </Button>
        )}
      </Card>
    </div>
  )
}

export default function ProtectedRoute({
  children,
  requiredPermissions = [],
  requiredRole,
  fallback,
  redirectTo = '/login'
}: ProtectedRouteProps) {
  const { isAuthenticated, isLoading, user } = useAuth()
  const { hasPermission, hasRole } = usePermissions()
  const router = useRouter()
  const pathname = usePathname()

  // Handle authentication redirect
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      // Encode the current path as return URL
      const returnUrl = encodeURIComponent(pathname)
      const loginUrl = `${redirectTo}?returnUrl=${returnUrl}`
      router.replace(loginUrl)
    }
  }, [isLoading, isAuthenticated, router, redirectTo, pathname])

  // Show loading state while authentication is being checked
  if (isLoading) {
    return fallback || <AuthLoading />
  }

  // If not authenticated, we'll be redirected by useEffect above
  // But show unauthorized in the meantime
  if (!isAuthenticated) {
    return (
      <Unauthorized
        message="Please sign in to access this page."
        showLoginButton
        onLoginRedirect={() => {
          const returnUrl = encodeURIComponent(pathname)
          const loginUrl = `${redirectTo}?returnUrl=${returnUrl}`
          router.push(loginUrl)
        }}
      />
    )
  }

  // Check role-based access
  if (requiredRole && !hasRole(requiredRole)) {
    return (
      <Unauthorized
        message={`This page requires ${requiredRole} role access.`}
      />
    )
  }

  // Check permission-based access
  const hasRequiredPermissions = requiredPermissions.length === 0 ||
    requiredPermissions.every(permission => hasPermission(permission))

  if (!hasRequiredPermissions) {
    const missingPermissions = requiredPermissions.filter(permission =>
      !hasPermission(permission)
    )

    return (
      <Unauthorized
        message={`This page requires additional permissions: ${missingPermissions.join(', ')}.`}
      />
    )
  }

  // User is authenticated and has required permissions
  return <>{children}</>
}

// Higher-order component for easier usage
export function withAuth<P extends object>(
  Component: React.ComponentType<P>,
  options: Omit<ProtectedRouteProps, 'children'> = {}
) {
  const WrappedComponent = (props: P) => (
    <ProtectedRoute {...options}>
      <Component {...props} />
    </ProtectedRoute>
  )

  WrappedComponent.displayName = `withAuth(${Component.displayName || Component.name})`

  return WrappedComponent
}

// Hook for imperatively checking auth in components
export function useAuthGuard(
  requiredPermissions: string[] = [],
  requiredRole?: string
) {
  const { isAuthenticated, isLoading, user } = useAuth()
  const { hasPermission, hasRole } = usePermissions()
  const router = useRouter()
  const pathname = usePathname()

  const checkAccess = () => {
    if (!isAuthenticated) {
      const returnUrl = encodeURIComponent(pathname)
      const loginUrl = `/login?returnUrl=${returnUrl}`
      router.replace(loginUrl)
      return false
    }

    if (requiredRole && !hasRole(requiredRole)) {
      return false
    }

    if (requiredPermissions.length > 0) {
      return requiredPermissions.every(permission => hasPermission(permission))
    }

    return true
  }

  return {
    isAuthenticated,
    isLoading,
    user,
    checkAccess,
    hasAccess: checkAccess(),
    canAccess: (permissions: string[] = [], role?: string) => {
      if (role && !hasRole(role)) return false
      return permissions.every(permission => hasPermission(permission))
    }
  }
}
