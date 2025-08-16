'use client'

import { useState, useEffect, useTransition, useRef } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useTranslations } from 'next-intl'
import Link from 'next/link'
import { IconShield, IconLoader2, IconRefresh, IconArrowLeft } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Alert } from '@/components/ui/alert'
import { useAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/use-toast'

export default function MFAPage() {
  const t = useTranslations('Auth')
  const router = useRouter()
  const searchParams = useSearchParams()
  const { verifyMFA, isLoading, error, clearError, isAuthenticated } = useAuth()
  const { toast } = useToast()
  const [isPending, startTransition] = useTransition()

  const [mfaCode, setMfaCode] = useState('')
  const [isResending, setIsResending] = useState(false)
  const [resendCooldown, setResendCooldown] = useState(0)
  const inputRefs = useRef<(HTMLInputElement | null)[]>([])

  // Redirect to dashboard if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      const returnUrl = searchParams.get('returnUrl') || '/dashboard'
      router.replace(returnUrl)
    }
  }, [isAuthenticated, router, searchParams])

  // Handle resend cooldown
  useEffect(() => {
    if (resendCooldown > 0) {
      const timer = setTimeout(() => {
        setResendCooldown(prev => prev - 1)
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [resendCooldown])

  // Clear errors when form changes
  useEffect(() => {
    if (error) {
      clearError()
    }
  }, [mfaCode, error, clearError])

  // Auto-focus first input on mount
  useEffect(() => {
    inputRefs.current[0]?.focus()
  }, [])

  const handleMfaCodeChange = (index: number, value: string) => {
    // Only allow digits
    if (!/^\d*$/.test(value)) return

    const newCode = mfaCode.split('')
    newCode[index] = value

    // Trim to 6 digits
    const updatedCode = newCode.slice(0, 6).join('')
    setMfaCode(updatedCode)

    // Auto-advance to next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus()
    }
  }

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !mfaCode[index] && index > 0) {
      inputRefs.current[index - 1]?.focus()
    } else if (e.key === 'ArrowLeft' && index > 0) {
      inputRefs.current[index - 1]?.focus()
    } else if (e.key === 'ArrowRight' && index < 5) {
      inputRefs.current[index + 1]?.focus()
    }
  }

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault()
    const pastedData = e.clipboardData.getData('text/plain').replace(/\D/g, '').slice(0, 6)
    setMfaCode(pastedData)

    // Focus the next empty input or the last input
    const nextIndex = Math.min(pastedData.length, 5)
    inputRefs.current[nextIndex]?.focus()
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (mfaCode.length !== 6) {
      toast({
        title: t('errors.mfaCodeLength'),
        variant: 'destructive',
      })
      return
    }

    startTransition(async () => {
      try {
        const result = await verifyMFA(mfaCode)

        if (result.success) {
          toast({
            title: t('loginSuccess'),
            description: t('loginSubtitle'),
            variant: 'default',
          })

          const returnUrl = searchParams.get('returnUrl') || '/dashboard'
          router.replace(returnUrl)
        }
      } catch (error) {
        console.error('MFA verification error:', error)
        toast({
          title: t('errors.mfaFailed'),
          description: t('errors.networkError'),
          variant: 'destructive',
        })
      }
    })
  }

  const handleResendCode = async () => {
    if (resendCooldown > 0 || isResending) return

    setIsResending(true)
    setResendCooldown(60) // 60 second cooldown

    try {
      // In a real implementation, this would call an API to resend the MFA code
      // For now, we'll just simulate the action
      await new Promise(resolve => setTimeout(resolve, 1000))

      toast({
        title: 'Code sent',
        description: 'A new verification code has been sent to your authenticator app',
        variant: 'default',
      })
    } catch (error) {
      toast({
        title: 'Failed to resend',
        description: 'Could not resend verification code. Please try again.',
        variant: 'destructive',
      })
      setResendCooldown(0)
    } finally {
      setIsResending(false)
    }
  }

  const renderMfaInputs = () => {
    const inputs = []

    for (let i = 0; i < 6; i++) {
      inputs.push(
        <Input
          key={i}
          ref={el => inputRefs.current[i] = el}
          type="text"
          inputMode="numeric"
          maxLength={1}
          value={mfaCode[i] || ''}
          onChange={e => handleMfaCodeChange(i, e.target.value)}
          onKeyDown={e => handleKeyDown(i, e)}
          onPaste={handlePaste}
          className="w-12 h-12 text-center text-lg font-mono border-2 focus:border-primary"
          disabled={isLoading || isPending}
          autoComplete="one-time-code"
          aria-label={`Digit ${i + 1} of verification code`}
        />
      )
    }

    return inputs
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/5 via-background to-secondary/5 p-4">
      <Card className="w-full max-w-md p-8 space-y-6 shadow-lg">
        <div className="text-center space-y-2">
          <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
            <IconShield className="h-6 w-6 text-primary" aria-hidden="true" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">
            {t('mfaTitle')}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t('mfaSubtitle')}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* MFA Code Inputs */}
          <div className="space-y-4">
            <label className="block text-sm font-medium text-foreground text-center">
              {t('mfaCode')}
            </label>
            <div className="flex justify-center space-x-2">
              {renderMfaInputs()}
            </div>
            <p className="text-xs text-muted-foreground text-center">
              {t('mfaCodePlaceholder')}
            </p>
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
            disabled={isLoading || isPending || mfaCode.length !== 6}
          >
            {(isLoading || isPending) ? (
              <>
                <IconLoader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                {t('verifying')}
              </>
            ) : (
              t('verifyButton')
            )}
          </Button>

          {/* Resend Code Button */}
          <div className="text-center">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleResendCode}
              disabled={resendCooldown > 0 || isResending}
              className="text-sm"
            >
              {isResending ? (
                <>
                  <IconLoader2 className="mr-2 h-3 w-3 animate-spin" aria-hidden="true" />
                  Sending...
                </>
              ) : resendCooldown > 0 ? (
                `${t('resendCode')} (${resendCooldown}s)`
              ) : (
                <>
                  <IconRefresh className="mr-2 h-3 w-3" aria-hidden="true" />
                  {t('resendCode')}
                </>
              )}
            </Button>
          </div>

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
