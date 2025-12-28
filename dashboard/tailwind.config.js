/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        platform: {
          reddit: '#FF4500',
          github: '#333333',
          wikipedia: '#636466',
          strava: '#FC4C02',
          lastfm: '#D51007',
          duolingo: '#58CC02',
          khan: '#14BF96',
          youtube: '#FF0000',
          twitter: '#1DA1F2',
          spotify: '#1DB954',
          goodreads: '#553B08',
          steam: '#171A21',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
