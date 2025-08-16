'use client'

import { useState, useEffect, useTransition } from 'react'
import { useRouter } from 'next/navigation'
import { useTranslations } from 'next-intl'
import Link from 'next/link'
import { IconMail, IconLoader2, IconArrowLeft, IconCheck } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Alert } from '@/components/ui/alert'
import { useAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/use-toast'

export default function PasswordResetPage() {
  const t = useTranslations('Auth')
  const router = useRouter()
  const { resetPassword, isAuthenticated } = useAuth()
  const { toast } = useToast()
  const [isPending, startTransition] = useTransition()

  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSuccess, setIsSuccess] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  // Redirect to dashboard if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      router.replace('/dashboard')
    }
  }, [isAuthenticated, router])

  // Clear errors when email changes
  useEffect(() => {
    setError(null)
    setValidationError(null)
  }, [email])

  const validateEmail = (): boolean => {
    if (!email) {
      setValidationError(t('errors.required'))
      return false
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setValidationError(t('errors.invalidEmail'))
      return false
    }

    setValidationError(null)
    return true
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateEmail()) return

    startTransition(async () => {
      try {
        const result = await resetPassword(email)

        if (result.success) {
          setIsSuccess(true)
          toast({
            title: t('passwordResetSent'),
            description: 'Please check your email for reset instructions',
            variant: 'default',
          })
        } else {
          setError(t('errors.resetFailed'))
        }
      } catch (error) {
        console.error('Password reset error:', error)
        setError(t('errors.networkError'))
      }
    })
  }

  const handleRetryReset = () => {
    setIsSuccess(false)
    setEmail('')
  }

  if (isSuccess) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/5 via-background to-secondary/5 p-4">
        <Card className="w-full max-w-md p-8 space-y-6 shadow-lg">
          <div className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
              <IconCheck className="h-8 w-8 text-green-600" aria-hidden="true" />
            </div>
            <div className="space-y-2">
              <h1 className="text-2xl font-bold text-foreground">
                Check your email
              </h1>
              <p className="text-sm text-muted-foreground">
                We've sent password reset instructions to:
              </p>
              <p className="text-sm font-medium text-foreground">
                {email}
              </p>
              <p className="text-xs text-muted-foreground mt-4">
                If you don't receive an email within a few minutes, check your spam folder or try again.
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <Button
              onClick={handleRetryReset}
              variant="outline"
              className="w-full"
            >
              Try different email
            </Button>

            <div className="text-center">
              <Link
                href="/login"
                className="inline-flex items-center text-sm text-primary hover:text-primary/80 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-sm"
              >
                <IconArrowLeft className="mr-1 h-3 w-3" aria-hidden="true" />
                {t('backToLogin')}
              </Link>
            </div>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/5 via-background to-secondary/5 p-4">
      <Card className="w-full max-w-md p-8 space-y-6 shadow-lg">
        <div className="text-center space-y-2">
          <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
            <IconMail className="h-6 w-6 text-primary" aria-hidden="true" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">
            {t('resetPasswordTitle')}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t('resetPasswordSubtitle')}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Email Field */}
          <div className="space-y-2">
            <label
              htmlFor="reset-email"
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
                id="reset-email"
                type="email"
                placeholder={t('emailPlaceholder')}
                value={email}
                onChange={e => setEmail(e.target.value)}
                className={`pl-10 ${validationError ? 'border-destructive focus:border-destructive' : ''}`}
                disabled={isPending}
                autoComplete="email"
                autoFocus
                aria-describedby={validationError ? 'email-error' : undefined}
              />
            </div>
            {validationError && (
              <p id="email-error" className="text-sm text-destructive" role="alert">
                {validationError}
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
            disabled={isPending || !email.trim()}
          >
            {isPending ? (
              <>
                <IconLoader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                Sending...
              </>
            ) : (
              t('resetPasswordButton')
            )}
          </Button>

          {/* Back to Login Link */}
          <div className="text-center">
            <Link
              href="/login"
              className="inline-flex items-center text-sm text-primary hover:text-primary/80 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-sm"
            >
              <IconArrowLeft className="mr-1 h-3 w-3" aria-hidden="true" />
              {t('backToLogin')}
            </Link>
          </div>
        </form>
      </Card>
    </div>
  )
}
