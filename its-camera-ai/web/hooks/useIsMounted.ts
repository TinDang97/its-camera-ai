'use client'

import { useCallback, useEffect, useRef } from 'react'

/**
 * Hook to check if component is still mounted
 * Prevents memory leaks by avoiding state updates on unmounted components
 */
export function useIsMounted(): () => boolean {
  const isMountedRef = useRef(true)

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  return useCallback(() => isMountedRef.current, [])
}
