/* Editorial, hand-drawn ornaments. Terracotta + olive on parchment —
   used where "empty state" would otherwise feel like a system error. */

export function PressedLeaf({ className = "w-20 h-20 opacity-80" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 140 140"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {/* midvein — slightly off-axis so it reads hand-drawn, not CAD */}
      <path d="M70 122 Q 68 80 73 22" stroke="#6B7A4F" strokeWidth="1" />
      {/* leaf blade */}
      <path
        d="M73 22 C 100 40, 104 84, 70 122 C 38 82, 48 38, 73 22 Z"
        fill="#fce6d6"
        fillOpacity="0.45"
        stroke="#c96442"
        strokeWidth="1.2"
      />
      {/* side veins — uneven lengths on purpose */}
      <path d="M70 50 Q 83 52 91 56" stroke="#6B7A4F" strokeWidth="0.7" />
      <path d="M70 50 Q 58 53 50 57"  stroke="#6B7A4F" strokeWidth="0.7" />
      <path d="M69 76 Q 85 79 93 83"  stroke="#6B7A4F" strokeWidth="0.7" />
      <path d="M69 76 Q 54 80 46 84"  stroke="#6B7A4F" strokeWidth="0.7" />
      <path d="M70 100 Q 80 102 85 103" stroke="#6B7A4F" strokeWidth="0.7" />
      <path d="M70 100 Q 60 102 55 103" stroke="#6B7A4F" strokeWidth="0.7" />
    </svg>
  );
}

/* Small ink-drop flourish — used as a section/chapter mark. */
export function InkMark({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <circle cx="12" cy="9"  r="3" fill="#c96442" />
      <circle cx="18" cy="14" r="1.4" fill="#6B7A4F" />
      <circle cx="7"  cy="16" r="1"   fill="#141413" />
    </svg>
  );
}
