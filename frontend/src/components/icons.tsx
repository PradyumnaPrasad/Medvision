// Small, consistent inline SVG icon set — replaces emoji throughout the app for a
// more clinical, professional look. All icons are 1em square and inherit color.

type IconProps = { className?: string };

export function AlertIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={2}>
      <path
        d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function UploadCloudIcon({ className = "w-8 h-8" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={1.6}>
      <path
        d="M7 18a4.5 4.5 0 0 1-.5-8.97A5.5 5.5 0 0 1 17.4 8.02 4 4 0 0 1 17 16h-1M12 11v8m0-8-3 3m3-3 3 3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ThumbsUpIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={1.8}>
      <path
        d="M7 10v11H4a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1h3Zm0 0 4.5-7.5a1.5 1.5 0 0 1 2.7.9L13.5 9H19a2 2 0 0 1 2 2.3l-1.3 8A2 2 0 0 1 17.72 21H10a3 3 0 0 1-3-3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ThumbsDownIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={1.8}>
      <path
        d="M17 14V3h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-3Zm0 0-4.5 7.5a1.5 1.5 0 0 1-2.7-.9L10.5 15H5a2 2 0 0 1-2-2.3l1.3-8A2 2 0 0 1 6.28 3H14a3 3 0 0 1 3 3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ChevronDownIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={2}>
      <path d="m6 9 6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function CheckCircleIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={2}>
      <path
        d="M9 12.5l2 2 4-4.5M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function XCircleIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={2}>
      <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM9.5 9.5l5 5m0-5-5 5" strokeLinecap="round" />
    </svg>
  );
}

export function GraphIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={1.8}>
      <circle cx="5" cy="6" r="2.4" />
      <circle cx="5" cy="18" r="2.4" />
      <circle cx="18" cy="12" r="2.8" />
      <path d="M7.1 7.1 15.6 11m-8.5 5.9L15.6 13" strokeLinecap="round" />
    </svg>
  );
}

export function LungsIcon({ className = "w-6 h-6" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className}>
      <path
        d="M9.5 5c-1.1 0-2 1.2-2 3v3.25c0 1-.3 1.5-.8 2.3C5.9 14.65 5 16.5 5 18.75 5 21.1 6.8 23 9.5 23c2 0 3-1.1 3-3v-4.5"
        stroke="#38bdf8"
        strokeWidth={1.6}
        strokeLinecap="round"
      />
      <path
        d="M14.5 5c1.1 0 2 1.2 2 3v3.25c0 1 .3 1.5.8 2.3.8 1.1 1.7 2.95 1.7 5.2 0 2.35-1.8 4.25-4.5 4.25-2 0-3-1.1-3-3v-4.5"
        stroke="#38bdf8"
        strokeWidth={1.6}
        strokeLinecap="round"
      />
      <circle cx="12" cy="6.5" r="1.6" fill="#4ade80" />
    </svg>
  );
}

export function DocumentIcon({ className = "w-4 h-4" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={1.8}>
      <path
        d="M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z M14 3v4h4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M9 12h6M9 15.5h6M9 8.5h2" strokeLinecap="round" />
    </svg>
  );
}

export function ClockIcon({ className = "w-3.5 h-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="1em" height="1em" className={className} stroke="currentColor" strokeWidth={1.8}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3.5 2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
