import type { Config } from "tailwindcss";

/* ------------------------------------------------------------------ *
 *  Design tokens — "Literary salon" system.
 *
 *  Philosophy: warm, quiet, editorial. Depth comes from background
 *  differentiation + 1px cream borders + ring halos, not heavy shadows.
 *  Every colour name encodes a role, not a hex — components consume
 *  roles, not literals.
 * ------------------------------------------------------------------ */

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        /* Page & surface colours — warmest at the bottom of the stack */
        paper: {
          DEFAULT: "#f5f4ed",  // Parchment — page background
          soft:    "#faf9f5",  // Ivory — cards, floating surfaces
          white:   "#ffffff",  // High-contrast occasional surface
          deep:    "#ede8db",  // Selection / hover tint
        },
        sand:    "#e8e6dc",    // Warm Sand — secondary button surface
        surface: {
          dark:   "#30302e",   // Dark surface (footer / dark sections)
          darker: "#141413",   // Near-black surface
        },

        /* Text — every grey has a warm yellow-brown undertone */
        ink: {
          DEFAULT: "#141413",  // Primary headings / UI chrome
          strong:  "#3d3d3a",  // Emphasised body (Dark Warm)
          body:    "#4d4c48",  // Paragraph text (Charcoal Warm)
          muted:   "#5e5d59",  // Secondary (Olive Gray)
          faint:   "#87867f",  // Tertiary / kickers (Stone Gray)
          silver:  "#b0aea5",  // On-dark-surface muted (Warm Silver)
        },

        /* Dividers, borders, halos — progression from whisper to firm */
        rule: {
          cream:   "#f0eee6",  // Border Cream — default card border
          DEFAULT: "#d9d5ca",  // Firm rule
          soft:    "#e7e3d6",  // Section divider
          warm:    "#e8e6dc",  // Alt border (shares hex with sand)
          hover:   "#d1cfc5",  // Ring Warm — hover/halo
          active:  "#c2c0b6",  // Ring Deep — pressed
        },

        /* Brand accent — terracotta is the ONLY colour that carries CTA weight */
        accent: {
          DEFAULT: "#c96442",  // Brand Terracotta
          coral:   "#d97757",  // Coral (lighter variant)
          soft:    "#fce6d6",  // Tint for badges / focus rings
          dark:    "#a84f35",  // Hover / pressed terracotta

          /* Chart / diagnostic secondaries — muted earth palette */
          teal:    "#3f5d56",
          olive:   "#6b7a4f",
          clay:    "#9b6a48",
        },

        /* Semantic tones — financial & callout surfaces */
        tone: {
          pos:  "#4f6b3e",
          warn: "#a07628",
          neg:  "#8b3a2b",
        },
        callout: {
          pos:     "#f1f3ea",
          warn:    "#f6efdd",
          neg:     "#f4e4df",
          neutral: "#ede8db",
        },

        /* System */
        error: "#b53333",      // Form validation (distinct from financial neg)
        focus: "#3898ec",      // Focus ring — the only cold colour allowed
      },

      fontFamily: {
        /* Sans: UI / body — MiSans unifies CN+EN with system fallbacks */
        sans: [
          '"MiSans"',
          '"PingFang SC"',
          '"Microsoft YaHei UI"',
          "Arial",
          "system-ui",
          "-apple-system",
          '"Segoe UI"',
          "Roboto",
          "sans-serif",
        ],
        /* Serif: headlines / pull quotes.
         * Per-character fallback resolves CJK glyphs through LXGW WenKai
         * (warm humanist kai) instead of Noto Serif SC's sharp 宋体 —
         * this keeps Latin editorial (Source Serif 4) while Chinese reads
         * as gentle, hand-written, matching the literary-salon tone. */
        serif: [
          '"Source Serif 4"',
          '"Source Serif Pro"',
          '"Tiempos Text"',
          "Georgia",
          '"Times New Roman"',
          "Times",
          '"LXGW WenKai"',
          '"LXGW WenKai Screen"',
          '"Noto Serif SC"',
          '"Songti SC"',
          "serif",
        ],
        /* Mono: code & tabular numerics */
        mono: [
          '"JetBrains Mono"',
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          '"Courier New"',
          "monospace",
        ],
      },

      /* Typographic scale.
       * The dashboard keeps compact sizes for dense tabular data (footnote/
       * caption/body-sm), but opens up generous editorial tokens for prose,
       * subtitles and section titles so the page reads like an article. */
      fontSize: {
        kicker:       ["11px",   { lineHeight: "1.2",  letterSpacing: "0.14em"   }],
        micro:        ["12px",   { lineHeight: "1.4",  letterSpacing: "0.01em"   }],
        footnote:     ["12.5px", { lineHeight: "1.5"                              }],
        caption:      ["13px",   { lineHeight: "1.45"                             }],
        "body-sm":    ["14px",   { lineHeight: "1.55"                             }],
        body:         ["16px",   { lineHeight: "1.6"                              }],
        "body-lg":    ["17px",   { lineHeight: "1.6"                              }],
        lede:         ["20px",   { lineHeight: "1.55"                             }],
        code:         ["15px",   { lineHeight: "1.6",  letterSpacing: "-0.015em" }],
        metric:       ["22px",   { lineHeight: "1"                                }],
        "metric-lg":  ["28px",   { lineHeight: "1"                                }],
        /* Editorial heading scale — serif, medium weight */
        "sub-sm":     ["25px",   { lineHeight: "1.2",  letterSpacing: "-0.003em" }],
        sub:          ["32px",   { lineHeight: "1.15", letterSpacing: "-0.006em" }],
        "sub-lg":     ["36px",   { lineHeight: "1.2",  letterSpacing: "-0.008em" }],
        section:      ["52px",   { lineHeight: "1.1",  letterSpacing: "-0.012em" }],
        display:      ["64px",   { lineHeight: "1.05", letterSpacing: "-0.015em" }],
        /* Legacy aliases — kept so older component strings don't break */
        "heading-sm": ["25px",   { lineHeight: "1.2",  letterSpacing: "-0.003em" }],
        "heading-lg": ["36px",   { lineHeight: "1.2",  letterSpacing: "-0.008em" }],
      },

      /* Radii — 8px-centric scale, matching the spec's 4/6/8/12/16/24/32.
       * Default is 12 so an un-suffixed `rounded` gives a warm, rounded card. */
      borderRadius: {
        xs:      "4px",
        "2xs":   "6px",
        sm:      "8px",   // Standard card / button
        md:      "12px",  // Input / primary button / key UI
        DEFAULT: "12px",
        lg:      "16px",  // Emphasised card / large container
        xl:      "24px",  // Feature container
        "2xl":   "32px",  // Hero / media
      },

      /* Elevation system.
       * L0: nothing → directly on parchment
       * L1: ring-1   → 1px cream containment (the default card)
       * L2: ring-2   → halo, used for hover/active hint
       * L3: whisper  → ring + very soft drop, floated just above the page
       * L4: pressed  → inset ring for active/depressed state
       * L5: elevated → ring + slightly stronger drop (modals / emphasis only)
       */
      boxShadow: {
        none:      "none",
        "ring-1":  "0 0 0 1px #f0eee6",
        "ring-2":  "0 0 0 1px #d1cfc5",
        whisper:   "0 0 0 1px #f0eee6, 0 4px 24px 0 rgba(20,20,19,0.04)",
        elevated:  "0 0 0 1px #e8e6dc, 0 6px 28px 0 rgba(20,20,19,0.06)",
        pressed:   "inset 0 0 0 1.5px #c2c0b6",
        focus:     "0 0 0 3px rgba(56,152,236,0.20)",
        "focus-warm": "0 0 0 3px rgba(201,100,66,0.18)",

        /* Legacy aliases */
        rule:      "inset 0 -1px 0 #d9d5ca",
        card:      "0 0 0 1px #f0eee6",
        "card-hover": "0 0 0 1px #d1cfc5, 0 2px 8px 0 rgba(20,20,19,0.05)",
        panel:     "0 0 0 1px #f0eee6, 0 4px 24px 0 rgba(20,20,19,0.04)",
        float:     "0 4px 24px 0 rgba(20,20,19,0.04)",
      },

      /* Content width — a columnar measure closer to a magazine spread */
      maxWidth: {
        prose:  "68ch",
        report: "1240px",
        wide:   "1440px",
      },
    },
  },
  plugins: [],
} satisfies Config;
