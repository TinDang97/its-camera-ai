'use client'

import React, { useState, useTransition, useCallback, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { IconEye, IconEyeOff, IconLoader2, IconShieldCheck, IconAlertTriangle } from '@tabler/icons-react'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Alert } from '@/components/ui/alert'

// Validation schema
const loginSchema = z.object({
  email: z
    .string()
    .min(1, 'Email is required')
    .email('Please enter a valid email address'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .max(128, 'Password must be less than 128 characters'),
  rememberMe: z.boolean().default(false),
})

type LoginFormData = z.infer<typeof loginSchema>

interface LoginFormProps {
  onMFARequired?: (tempToken: string) => void
  onSuccess?: () => void
  redirectTo?: string
  className?: string
}

export default function LoginForm({
  onMFARequired,
  onSuccess,
  redirectTo,
  className = ""
}: LoginFormProps) {
  const [isPending, startTransition] = useTransition()
  const [showPassword, setShowPassword] = useState(false)
  const [loginError, setLoginError] = useState<string | null>(null)
  const [requiresMFA, setRequiresMFA] = useState(false)

  const { login, isLoading: authLoading, error: authError } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()

  // Get return URL from query params or use provided redirectTo
  const returnUrl = searchParams.get('returnUrl') || redirectTo || '/dashboard'

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    setError,
    clearErrors,
    getValues,
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
      rememberMe: false,
    }
  })

  // Clear errors when form changes
  useEffect(() => {
    if (loginError || authError) {
      const timeoutId = setTimeout(() => {
        setLoginError(null)
        clearErrors()
      }, 5000)
      return () => clearTimeout(timeoutId)
    }
  }, [loginError, authError, clearErrors])

  // Handle form submission
  const onSubmit = useCallback(async (data: LoginFormData) => {
    startTransition(async () => {
      try {
        setLoginError(null)
        clearErrors()

        // Store remember me preference
        if (data.rememberMe) {
          localStorage.setItem('auth_remember_me', 'true')
        } else {
          localStorage.removeItem('auth_remember_me')
        }

        // Attempt login
        const result = await login(data.email, data.password)

        if (result.success) {
          onSuccess?.()
          // Redirect on successful login
          router.replace(decodeURIComponent(returnUrl))
        } else if (result.requiresMFA) {
          setRequiresMFA(true)
          onMFARequired?.(data.email) // Pass email for MFA context
        }
      } catch (error) {
        console.error('Login failed:', error)

        // Handle specific error types
        if (error instanceof Error) {
          if (error.message.includes('Invalid credentials')) {
            setError('password', {
              message: 'Invalid email or password. Please try again.'
            })
          } else if (error.message.includes('Account locked')) {
            setLoginError('Account temporarily locked due to too many failed attempts. Please try again later.')
          } else if (error.message.includes('Email not verified')) {
            setLoginError('Please verify your email address before signing in.')
          } else {
            setLoginError(error.message || 'Login failed. Please try again.')
          }
        } else {
          setLoginError('An unexpected error occurred. Please try again.')
        }
      }
    })
  }, [login, onSuccess, onMFARequired, router, returnUrl, clearErrors, setError])

  // Toggle password visibility
  const togglePasswordVisibility = useCallback(() => {
    setShowPassword(prev => !prev)
  }, [])

  // Loading state
  const isLoading = isPending || authLoading || isSubmitting

  return (
    <Card className={`w-full max-w-md mx-auto p-8 space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
          <IconShieldCheck className="h-8 w-8 text-primary" aria-hidden="true" />
        </div>
        <h1 className="text-2xl font-bold text-foreground">
          Sign In
        </h1>
        <p className="text-sm text-muted-foreground">
          Access your ITS Camera AI dashboard
        </p>
      </div>

      {/* MFA Notice */}
      {requiresMFA && (
        <Alert className="border-secondary/20 bg-secondary/5">
          <IconShieldCheck className="h-4 w-4 text-secondary" />
          <div className="space-y-1">
            <p className="font-medium text-secondary">Multi-Factor Authentication Required</p>
            <p className="text-sm text-muted-foreground">
              Please check your authenticator app for the verification code.
            </p>
          </div>
        </Alert>
      )}

      {/* Error Display */}
      {(loginError || authError) && (
        <Alert variant="destructive" className="animate-in slide-in-from-top-2">
          <IconAlertTriangle className="h-4 w-4" />
          <p className="text-sm font-medium">
            {loginError || authError}
          </p>
        </Alert>
      )}

      {/* Login Form */}
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4" noValidate>
        {/* Email Field */}
        <div className="space-y-2">
          <label
            htmlFor="email"
            className="text-sm font-medium text-foreground"
          >
            Email Address
          </label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            disabled={isLoading}
            className={errors.email ? 'border-destructive focus-visible:ring-destructive' : ''}
            placeholder="Enter your email address"
            aria-invalid={errors.email ? 'true' : 'false'}
            aria-describedby={errors.email ? 'email-error' : undefined}
            {...register('email')}
          />
          {errors.email && (
            <p
              id="email-error"
              className="text-sm text-destructive animate-in slide-in-from-top-1"
              role="alert"
              aria-live="polite"
            >
              {errors.email.message}
            </p>
          )}
        </div>

        {/* Password Field */}
        <div className="space-y-2">
          <label
            htmlFor="password"
            className="text-sm font-medium text-foreground"
          >
            Password
          </label>
          <div className="relative">
            <Input
              id="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              disabled={isLoading}
              className={`pr-10 ${errors.password ? 'border-destructive focus-visible:ring-destructive' : ''}`}
              placeholder="Enter your password"
              aria-invalid={errors.password ? 'true' : 'false'}
              aria-describedby={errors.password ? 'password-error' : undefined}
              {...register('password')}
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
              onClick={togglePasswordVisibility}
              disabled={isLoading}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? (
                <IconEyeOff className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              ) : (
                <IconEye className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              )}
            </Button>
          </div>
          {errors.password && (
            <p
              id="password-error"
              className="text-sm text-destructive animate-in slide-in-from-top-1"
              role="alert"
              aria-live="polite"
            >
              {errors.password.message}
            </p>
          )}
        </div>

        {/* Remember Me Checkbox */}
        <div className="flex items-center space-x-2">
          <input
            id="rememberMe"
            type="checkbox"
            disabled={isLoading}
            className="h-4 w-4 rounded border-input text-primary focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50"
            {...register('rememberMe')}
          />
          <label
            htmlFor="rememberMe"
            className="text-sm font-medium text-foreground cursor-pointer"
          >
            Keep me signed in for 30 days
          </label>
        </div>

        {/* Submit Button */}
        <Button
          type="submit"
          className="w-full"
          disabled={isLoading}
          aria-describedby={isLoading ? 'loading-message' : undefined}
        >
          {isLoading ? (
            <>
              <IconLoader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
              <span id="loading-message">Signing in...</span>
            </>
          ) : (
            'Sign In'
          )}
        </Button>
      </form>

      {/* Footer Links */}
      <div className="text-center space-y-2">
        <Button
          variant="link"
          size="sm"
          className="text-sm text-muted-foreground hover:text-foreground"
          onClick={() => router.push('/auth/forgot-password')}
          disabled={isLoading}
        >
          Forgot your password?
        </Button>

        <div className="text-xs text-muted-foreground">
          Need help? Contact{' '}
          <a
            href="mailto:support@itscameraai.com"
            className="text-primary hover:underline focus:underline focus:outline-none"
            tabIndex={0}
          >
            support@itscameraai.com
          </a>
        </div>
      </div>
    </Card>
  )
}
