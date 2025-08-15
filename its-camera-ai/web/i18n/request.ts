import {getRequestConfig} from 'next-intl/server';
import {routing} from './config';

export default getRequestConfig(async ({requestLocale}) => {
  // This typically corresponds to the `[locale]` segment
  let locale = await requestLocale;

  // Ensure that the incoming locale is valid
  if (!locale || !routing.locales.includes(locale as any)) {
    locale = routing.defaultLocale;
  }

  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
    // Configure date/time formatting
    timeZone: 'UTC',
    // Configure number formatting
    formats: {
      dateTime: {
        short: {
          day: 'numeric',
          month: 'short',
          year: 'numeric'
        },
        medium: {
          day: 'numeric',
          month: 'long',
          year: 'numeric'
        },
        long: {
          day: 'numeric',
          month: 'long',
          year: 'numeric',
          weekday: 'long'
        },
        time: {
          hour: 'numeric',
          minute: 'numeric'
        },
        datetime: {
          day: 'numeric',
          month: 'short',
          year: 'numeric',
          hour: 'numeric',
          minute: 'numeric'
        }
      },
      number: {
        precise: {
          maximumFractionDigits: 3
        },
        percentage: {
          style: 'percent',
          maximumFractionDigits: 1
        }
      }
    }
  };
});
