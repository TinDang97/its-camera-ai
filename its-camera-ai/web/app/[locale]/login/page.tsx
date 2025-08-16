'use client'

import { useState, useEffect, useTransition } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useTranslations } from 'next-intl'
import Link from 'next/link'
import { IconEye, IconEyeOff, IconLoader2, IconLock, IconMail } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Alert } from '@/components/ui/alert'
import { useAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/use-toast'

interface LoginFormData {
  email: string
  password: string
}

interface ValidationErrors {
  email?: string
  password?: string
}

export default function LoginPage() {
  const t = useTranslations('Auth')
  const router = useRouter()
  const searchParams = useSearchParams()
  const { login, isLoading, error, clearError, isAuthenticated } = useAuth()
  const { toast } = useToast()
  const [isPending, startTransition] = useTransition()

  const [formData, setFormData] = useState<LoginFormData>({
    email: '',
    password: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({})
  const [requiresMFA, setRequiresMFA] = useState(false)

  // Redirect to dashboard if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      const returnUrl = searchParams.get('returnUrl') || '/dashboard'
      router.replace(returnUrl)
    }
  }, [isAuthenticated, router, searchParams])

  // Clear errors when form changes
  useEffect(() => {
    if (error) {
      clearError()
    }
    setValidationErrors({})
  }, [formData, error, clearError])

  const validateForm = (): boolean => {
    const errors: ValidationErrors = {}

    if (!formData.email) {
      errors.email = t('errors.required')
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = t('errors.invalidEmail')
    }

    if (!formData.password) {
      errors.password = t('errors.required')
    } else if (formData.password.length < 8) {
      errors.password = t('errors.passwordMinLength')
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleInputChange = (field: keyof LoginFormData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({
      ...prev,
      [field]: e.target.value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) return

    startTransition(async () => {
      try {
        const result = await login(formData.email, formData.password)

        if (result.success) {
          toast({
            title: t('loginSuccess'),
            description: t('loginSubtitle'),
            variant: 'default',
          })

          const returnUrl = searchParams.get('returnUrl') || '/dashboard'
          router.replace(returnUrl)
        } else if (result.requiresMFA) {
          setRequiresMFA(true)
          // Redirect to MFA page while preserving return URL
          const returnUrl = searchParams.get('returnUrl') || '/dashboard'
          router.push(`/login/mfa?returnUrl=${encodeURIComponent(returnUrl)}`)
        }
      } catch (error) {
        console.error('Login error:', error)
        toast({
          title: t('errors.loginFailed'),
          description: t('errors.networkError'),
          variant: 'destructive',
        })
      }
    })
  }

  const togglePasswordVisibility = () => {
    setShowPassword(prev => !prev)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/5 via-background to-secondary/5 p-4">
      <Card className="w-full max-w-md p-8 space-y-6 shadow-lg">
        <div className="text-center space-y-2">
          <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
            <IconLock className="h-6 w-6 text-primary" aria-hidden="true" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">
            {t('loginTitle')}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t('loginSubtitle')}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Email Field */}
          <div className="space-y-2">
            <label
              htmlFor="email"
              className="text-sm font-medium text-foreground"
            >
              {t('email')}
            </label>
            <div className="relative">
              <IconMail
                className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                id="email"
                type="email"
                placeholder={t('emailPlaceholder')}
                value={formData.email}
                onChange={handleInputChange('email')}
                className={`pl-10 ${validationErrors.email ? 'border-destructive focus:border-destructive' : ''}`}
                disabled={isLoading || isPending}
                autoComplete="email"
                aria-describedby={validationErrors.email ? 'email-error' : undefined}
              />
            </div>
            {validationErrors.email && (
              <p id="email-error" className="text-sm text-destructive" role="alert">
                {validationErrors.email}
              </p>
            )}
          </div>

          {/* Password Field */}
          <div className="space-y-2">
            <label
              htmlFor="password"
              className="text-sm font-medium text-foreground"
            >
              {t('password')}
            </label>
            <div className="relative">
              <IconLock
                className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                id="password"
                type={showPassword ? 'text' : 'password'}
                placeholder={t('passwordPlaceholder')}
                value={formData.password}
                onChange={handleInputChange('password')}
                className={`pl-10 pr-10 ${validationErrors.password ? 'border-destructive focus:border-destructive' : ''}`}
                disabled={isLoading || isPending}
                autoComplete="current-password"
                aria-describedby={validationErrors.password ? 'password-error' : undefined}
              />
              <button
                type="button"
                onClick={togglePasswordVisibility}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground focus:outline-none focus:text-foreground"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <IconEyeOff className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <IconEye className="h-4 w-4" aria-hidden="true" />
                )}
              </button>
            </div>
            {validationErrors.password && (
              <p id="password-error" className="text-sm text-destructive" role="alert">
                {validationErrors.password}
              </p>
            )}
          </div>

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive" role="alert">
              <p>{error}</p>
            </Alert>
          )}

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full"
            disabled={isLoading || isPending}
          >
            {(isLoading || isPending) ? (
              <>
                <IconLoader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                {t('loggingIn')}
              </>
            ) : (
              t('loginButton')
            )}
          </Button>

          {/* Forgot Password Link */}
          <div className="text-center">
            <Link
              href="/login/reset"
              className="text-sm text-primary hover:text-primary/80 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-sm"
            >
              {t('forgotPassword')}
            </Link>
          </div>
        </form>
      </Card>
    </div>
  )
}
