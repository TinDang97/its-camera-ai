'use client'

import React, { useState, useRef, useCallback, useEffect, useOptimistic, useTransition } from 'react'
import { useRouter } from 'next/navigation'
import { IconShieldCheck, IconLoader2, IconAlertTriangle, IconRefresh, IconArrowLeft } from '@tabler/icons-react'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Alert } from '@/components/ui/alert'

interface MFAVerificationProps {
  email?: string
  onBack?: () => void
  onSuccess?: () => void
  redirectTo?: string
  className?: string
}

interface MFAState {
  code: string
  isVerifying: boolean
  error: string | null
  resendCooldown: number
  resendAttempts: number
}

const INITIAL_STATE: MFAState = {
  code: '',
  isVerifying: false,
  error: null,
  resendCooldown: 0,
  resendAttempts: 0,
}

const RESEND_COOLDOWN_SECONDS = 30
const MAX_RESEND_ATTEMPTS = 3
const CODE_LENGTH = 6

export default function MFAVerification({
  email,
  onBack,
  onSuccess,
  redirectTo = '/dashboard',
  className = ""
}: MFAVerificationProps) {
  const [isPending, startTransition] = useTransition()
  const [state, setState] = useState<MFAState>(INITIAL_STATE)
  const [optimisticState, updateOptimisticState] = useOptimistic(state)
  const inputRefs = useRef<(HTMLInputElement | null)[]>([])
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const { verifyMFA, isLoading: authLoading } = useAuth()
  const router = useRouter()

  // Initialize input refs
  useEffect(() => {
    inputRefs.current = inputRefs.current.slice(0, CODE_LENGTH)
  }, [])

  // Handle resend cooldown timer
  useEffect(() => {
    if (state.resendCooldown > 0) {
      intervalRef.current = setInterval(() => {
        setState(prev => ({
          ...prev,
          resendCooldown: Math.max(0, prev.resendCooldown - 1)
        }))
      }, 1000)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [state.resendCooldown])

  // Auto-focus first input on mount
  useEffect(() => {
    inputRefs.current[0]?.focus()
  }, [])

  // Handle input change with auto-progression
  const handleInputChange = useCallback((index: number, value: string) => {
    // Only allow digits
    const numericValue = value.replace(/[^0-9]/g, '')

    if (numericValue.length > 1) {
      // Handle paste operation
      const pastedDigits = numericValue.slice(0, CODE_LENGTH)
      const newCode = pastedDigits.padEnd(CODE_LENGTH, '').split('')

      // Update all inputs
      inputRefs.current.forEach((input, i) => {
        if (input) {
          input.value = newCode[i] || ''
        }
      })

      // Focus on the next empty input or the last input
      const nextEmptyIndex = newCode.findIndex(digit => !digit)
      const focusIndex = nextEmptyIndex === -1 ? CODE_LENGTH - 1 : nextEmptyIndex
      inputRefs.current[focusIndex]?.focus()

      // Update state
      setState(prev => ({
        ...prev,
        code: newCode.join(''),
        error: null
      }))
    } else {
      // Single character input
      const newCode = state.code.split('')
      newCode[index] = numericValue

      const updatedCode = newCode.join('')
      setState(prev => ({
        ...prev,
        code: updatedCode,
        error: null
      }))

      // Auto-advance to next input
      if (numericValue && index < CODE_LENGTH - 1) {
        inputRefs.current[index + 1]?.focus()
      }
    }
  }, [state.code])

  // Handle backspace/delete key
  const handleKeyDown = useCallback((index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' || e.key === 'Delete') {
      e.preventDefault()

      const newCode = state.code.split('')
      newCode[index] = ''

      setState(prev => ({
        ...prev,
        code: newCode.join(''),
        error: null
      }))

      // Move focus to previous input if current is empty
      if (!state.code[index] && index > 0) {
        inputRefs.current[index - 1]?.focus()
      }
    } else if (e.key === 'ArrowLeft' && index > 0) {
      inputRefs.current[index - 1]?.focus()
    } else if (e.key === 'ArrowRight' && index < CODE_LENGTH - 1) {
      inputRefs.current[index + 1]?.focus()
    }
  }, [state.code])

  // Verify MFA code
  const handleVerifyCode = useCallback(async () => {
    if (optimisticState.code.length !== CODE_LENGTH) {
      setState(prev => ({ ...prev, error: 'Please enter the complete 6-digit code.' }))
      return
    }

    startTransition(async () => {
      // Optimistic update
      updateOptimisticState(prev => ({
        ...prev,
        isVerifying: true,
        error: null
      }))

      try {
        const result = await verifyMFA(optimisticState.code)

        if (result.success) {
          onSuccess?.()
          router.replace(redirectTo)
        }
      } catch (error) {
        console.error('MFA verification failed:', error)

        let errorMessage = 'Verification failed. Please try again.'
        if (error instanceof Error) {
          if (error.message.includes('Invalid code')) {
            errorMessage = 'Invalid verification code. Please check your authenticator app.'
          } else if (error.message.includes('Code expired')) {
            errorMessage = 'Verification code has expired. Please request a new one.'
          } else if (error.message.includes('Too many attempts')) {
            errorMessage = 'Too many failed attempts. Please wait before trying again.'
          }
        }

        setState(prev => ({
          ...prev,
          isVerifying: false,
          error: errorMessage,
          code: '' // Clear code on error
        }))

        // Clear all inputs and focus first one
        inputRefs.current.forEach(input => {
          if (input) input.value = ''
        })
        inputRefs.current[0]?.focus()
      }
    })
  }, [optimisticState.code, verifyMFA, onSuccess, router, redirectTo, updateOptimisticState])

  // Handle resend code
  const handleResendCode = useCallback(async () => {
    if (state.resendCooldown > 0 || state.resendAttempts >= MAX_RESEND_ATTEMPTS) {
      return
    }

    startTransition(async () => {
      try {
        // This would call a resend endpoint
        // await apiClient.request(ENDPOINTS.AUTH.MFA.RESEND, { method: 'POST' })

        setState(prev => ({
          ...prev,
          resendCooldown: RESEND_COOLDOWN_SECONDS,
          resendAttempts: prev.resendAttempts + 1,
          error: null
        }))
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Failed to resend verification code. Please try again.'
        }))
      }
    })
  }, [state.resendCooldown, state.resendAttempts])

  // Auto-submit when code is complete
  useEffect(() => {
    if (state.code.length === CODE_LENGTH && !state.isVerifying) {
      handleVerifyCode()
    }
  }, [state.code, state.isVerifying, handleVerifyCode])

  const isLoading = isPending || authLoading || optimisticState.isVerifying

  return (
    <Card className={`w-full max-w-md mx-auto p-8 space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="mx-auto w-16 h-16 bg-secondary/10 rounded-full flex items-center justify-center">
          <IconShieldCheck className="h-8 w-8 text-secondary" aria-hidden="true" />
        </div>
        <h1 className="text-2xl font-bold text-foreground">
          Two-Factor Authentication
        </h1>
        <div className="text-sm text-muted-foreground space-y-1">
          <p>Enter the 6-digit code from your authenticator app</p>
          {email && (
            <p className="font-medium text-foreground">
              Sent to: {email.replace(/(.{2}).*@/, '$1***@')}
            </p>
          )}
        </div>
      </div>

      {/* Error Display */}
      {state.error && (
        <Alert variant="destructive" className="animate-in slide-in-from-top-2">
          <IconAlertTriangle className="h-4 w-4" />
          <p className="text-sm font-medium">{state.error}</p>
        </Alert>
      )}

      {/* Code Input Grid */}
      <div className="space-y-4">
        <div
          className="flex justify-center space-x-2"
          role="group"
          aria-label="Enter 6-digit verification code"
        >
          {Array.from({ length: CODE_LENGTH }, (_, index) => (
            <input
              key={index}
              ref={(el) => (inputRefs.current[index] = el)}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={CODE_LENGTH} // Allow paste
              disabled={isLoading}
              className="w-12 h-12 text-center text-lg font-mono font-bold border border-input rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label={`Digit ${index + 1} of 6`}
              onChange={(e) => handleInputChange(index, e.target.value)}
              onKeyDown={(e) => handleKeyDown(index, e)}
              autoComplete="one-time-code"
            />
          ))}
        </div>

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
            <IconLoader2 className="h-4 w-4 animate-spin" />
            <span>Verifying code...</span>
          </div>
        )}
      </div>

      {/* Resend Section */}
      <div className="text-center space-y-2">
        {state.resendCooldown > 0 ? (
          <p className="text-sm text-muted-foreground">
            Resend code in {state.resendCooldown} seconds
          </p>
        ) : state.resendAttempts >= MAX_RESEND_ATTEMPTS ? (
          <p className="text-sm text-destructive">
            Maximum resend attempts reached. Please contact support.
          </p>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleResendCode}
            disabled={isLoading}
            className="text-primary hover:text-primary-hover"
          >
            <IconRefresh className="mr-2 h-4 w-4" />
            Resend verification code
          </Button>
        )}

        {state.resendAttempts > 0 && state.resendAttempts < MAX_RESEND_ATTEMPTS && (
          <p className="text-xs text-muted-foreground">
            {MAX_RESEND_ATTEMPTS - state.resendAttempts} resend attempts remaining
          </p>
        )}
      </div>

      {/* Back Button */}
      {onBack && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          disabled={isLoading}
          className="w-full mt-4"
        >
          <IconArrowLeft className="mr-2 h-4 w-4" />
          Back to sign in
        </Button>
      )}

      {/* Help Text */}
      <div className="text-center text-xs text-muted-foreground">
        <p>Having trouble? Check that your device's time is correct.</p>
        <p className="mt-1">
          Need help?{' '}
          <a
            href="mailto:support@itscameraai.com"
            className="text-primary hover:underline focus:underline focus:outline-none"
          >
            Contact support
          </a>
        </p>
      </div>
    </Card>
  )
}
