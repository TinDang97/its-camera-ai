'use client'

import { useState, useEffect, useCallback, useContext, createContext, ReactNode } from 'react'
import { authUtils, apiClient, AuthTokens, APIError, ENDPOINTS } from '@/lib/api'
import { useAuthState, useRealtimeActions } from '@/stores/useRealtimeStore'

export interface User {
  id: string
  email: string
  name: string
  role: string
  permissions: string[]
  is_active: boolean
  is_verified: boolean
  last_login?: string
  created_at: string
  updated_at: string
}

interface AuthState {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  error: string | null
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<{ success: boolean; requiresMFA?: boolean }>
  logout: () => Promise<void>
  refreshUser: () => Promise<void>
  clearError: () => void
  verifyMFA: (code: string) => Promise<{ success: boolean }>
  resetPassword: (email: string) => Promise<{ success: boolean }>
}

const AuthContext = createContext<AuthContextType | null>(null)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const authState = useAuthState()
  const { setUser, setAuthLoading, setAuthError, clearAuthError: clearStoreError } = useRealtimeActions()

  // Local state for compatibility with existing context API
  const [localError, setLocalError] = useState<string | null>(null)

  const state: AuthState = {
    user: authState.user,
    isLoading: authState.isAuthLoading,
    isAuthenticated: authState.isAuthenticated,
    error: authState.authError || localError,
  }

  const clearError = useCallback(() => {
    clearStoreError()
    setLocalError(null)
  }, [clearStoreError])

  const refreshUser = useCallback(async () => {
    if (!authUtils.isAuthenticated()) {
      setUser(null)
      setAuthLoading(false)
      return
    }

    try {
      setAuthLoading(true)
      clearError()
      const user = await authUtils.getCurrentUser()
      setUser(user)
    } catch (error) {
      console.error('Failed to refresh user:', error)
      setUser(null)
      const errorMessage = error instanceof APIError ? error.message : 'Failed to authenticate'
      setAuthError(errorMessage)
    }
  }, [setUser, setAuthLoading, setAuthError, clearError])

  const login = useCallback(async (email: string, password: string) => {
    try {
      setAuthLoading(true)
      clearError()

      const tokens = await authUtils.login(email, password)

      // Check if MFA is required (tokens will be temporary)
      if (tokens.token_type === 'mfa_required') {
        setAuthLoading(false)
        return { success: false, requiresMFA: true }
      }

      // Login successful, refresh user data
      await refreshUser()
      return { success: true }
    } catch (error) {
      console.error('Login failed:', error)
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Login failed. Please check your credentials.'

      setAuthError(errorMessage)
      return { success: false }
    }
  }, [refreshUser, setAuthLoading, setAuthError, clearError])

  const verifyMFA = useCallback(async (code: string) => {
    try {
      setAuthLoading(true)
      clearError()

      // Call MFA verification endpoint
      await apiClient.request<AuthTokens>(ENDPOINTS.AUTH.MFA.VERIFY, {
        method: 'POST',
        body: JSON.stringify({ code }),
      })

      // MFA successful, refresh user data
      await refreshUser()
      return { success: true }
    } catch (error) {
      console.error('MFA verification failed:', error)
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Invalid MFA code. Please try again.'

      setAuthError(errorMessage)
      return { success: false }
    }
  }, [refreshUser, setAuthLoading, setAuthError, clearError])

  const logout = useCallback(async () => {
    try {
      setAuthLoading(true)
      clearError()
      await authUtils.logout()
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      setUser(null)
      clearError()
    }
  }, [setUser, setAuthLoading, clearError])

  const resetPassword = useCallback(async (email: string) => {
    try {
      await apiClient.request(ENDPOINTS.AUTH.RESET_PASSWORD, {
        method: 'POST',
        body: JSON.stringify({ email }),
        skipAuth: true,
      })
      return { success: true }
    } catch (error) {
      console.error('Password reset failed:', error)
      return { success: false }
    }
  }, [])

  // Initialize authentication state on mount
  useEffect(() => {
    refreshUser()
  }, [refreshUser])

  // Listen for storage events to sync auth state across tabs
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'auth_tokens') {
        if (e.newValue === null) {
          // Tokens were cleared in another tab
          setUser(null)
          clearError()
        } else if (e.oldValue === null && e.newValue) {
          // User logged in another tab
          refreshUser()
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [refreshUser, setUser, clearError])

  const value: AuthContextType = {
    ...state,
    login,
    logout,
    refreshUser,
    clearError,
    verifyMFA,
    resetPassword,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Hook for protected routes
export function useRequireAuth(redirectTo = '/login') {
  const { isAuthenticated, isLoading } = useAuth()

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      window.location.href = redirectTo
    }
  }, [isAuthenticated, isLoading, redirectTo])

  return { isAuthenticated, isLoading }
}

// Hook for role-based access control
export function usePermissions() {
  const { user } = useAuth()

  const hasPermission = useCallback((permission: string) => {
    return user?.permissions?.includes(permission) ?? false
  }, [user])

  const hasRole = useCallback((role: string) => {
    return user?.role === role
  }, [user])

  const hasAnyRole = useCallback((roles: string[]) => {
    return roles.includes(user?.role ?? '')
  }, [user])

  return {
    hasPermission,
    hasRole,
    hasAnyRole,
    permissions: user?.permissions ?? [],
    role: user?.role,
  }
}
