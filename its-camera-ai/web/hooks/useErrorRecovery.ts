'use client';

import { useState, useCallback, useRef } from 'react';

interface ErrorRecoveryState {
  hasError: boolean;
  errorCount: number;
  lastError: Error | null;
  context?: string;
}

interface UseErrorRecoveryOptions {
  maxRetries?: number;
  retryDelay?: number;
  exponentialBackoff?: boolean;
}

export function useErrorRecovery(options: UseErrorRecoveryOptions = {}) {
  const {
    maxRetries = 3,
    retryDelay = 1000,
    exponentialBackoff = true,
  } = options;

  const [state, setState] = useState<ErrorRecoveryState>({
    hasError: false,
    errorCount: 0,
    lastError: null,
  });

  const retryTimeoutRef = useRef<NodeJS.Timeout>();

  const reportError = useCallback((error: Error, context = 'operation') => {
    setState(prev => ({
      hasError: true,
      errorCount: prev.errorCount + 1,
      lastError: error,
      context,
    }));
  }, []);

  const clearError = useCallback(() => {
    setState({
      hasError: false,
      errorCount: 0,
      lastError: null,
    });

    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
    }
  }, []);

  const retryOperation = useCallback(async <T>(
    operation: () => Promise<T>,
    context = 'retry'
  ): Promise<T | null> => {
    if (state.errorCount >= maxRetries) {
      console.error(`Max retries (${maxRetries}) exceeded for ${context}`);
      return null;
    }

    const delay = exponentialBackoff
      ? retryDelay * Math.pow(2, state.errorCount)
      : retryDelay;

    return new Promise((resolve) => {
      retryTimeoutRef.current = setTimeout(async () => {
        try {
          const result = await operation();
          clearError();
          resolve(result);
        } catch (error) {
          reportError(error as Error, context);
          resolve(null);
        }
      }, delay);
    });
  }, [state.errorCount, maxRetries, retryDelay, exponentialBackoff, clearError, reportError]);

  const executeWithRecovery = useCallback(async <T>(
    operation: () => Promise<T>,
    context = 'operation'
  ): Promise<T | null> => {
    try {
      clearError();
      const result = await operation();
      return result;
    } catch (error) {
      const err = error as Error;
      reportError(err, context);

      // Auto-retry for network errors
      if (err.message.includes('fetch') || err.message.includes('network')) {
        console.log(`Network error detected, attempting retry for ${context}`);
        return retryOperation(operation, context);
      }

      throw err;
    }
  }, [clearError, reportError, retryOperation]);

  const isRecoverable = useCallback((error: Error) => {
    // Define which errors are recoverable
    const recoverableErrors = [
      'NetworkError',
      'TimeoutError',
      'fetch',
      'network',
      'connection',
    ];

    return recoverableErrors.some(pattern =>
      error.name.includes(pattern) || error.message.toLowerCase().includes(pattern)
    );
  }, []);

  return {
    ...state,
    executeWithRecovery,
    retryOperation,
    reportError,
    clearError,
    isRecoverable,
    canRetry: state.errorCount < maxRetries,
  };
}
