"""
The asymmetric confidence gauge — the same visual signature used in the
companion HTML demo. Rendered as inline SVG so it works identically across
both deliverables.
"""

def gauge_svg(prob, width_pct=100):
    W, H, x0, x1, y = 600, 120, 40, 560, 55
    span = x1 - x0

    def x_at(p):
        return x0 + span * p

    sell_end = x_at(0.48)
    buy_start = x_at(0.60)
    needle_x = x_at(min(1, max(0, prob)))

    return f"""
    <svg viewBox="0 0 {W} {H}" style="width:{width_pct}%; height:auto; display:block;"
         role="img" aria-label="Predicted probability of an up move: {prob*100:.1f} percent">
      <line x1="{x0}" y1="{y}" x2="{sell_end}" y2="{y}" stroke="#E07A4F" stroke-width="18" />
      <line x1="{sell_end}" y1="{y}" x2="{buy_start}" y2="{y}" stroke="#E0B83A" stroke-width="18" />
      <line x1="{buy_start}" y1="{y}" x2="{x1}" y2="{y}" stroke="#3FB39E" stroke-width="18" />
      <text x="{(x0+sell_end)/2}" y="34" font-family="IBM Plex Mono, monospace" font-size="11"
            fill="#9FB3AC" text-anchor="middle" letter-spacing="1">SELL ZONE</text>
      <text x="{(sell_end+buy_start)/2}" y="34" font-family="IBM Plex Mono, monospace" font-size="11"
            fill="#9FB3AC" text-anchor="middle" letter-spacing="1">HOLD</text>
      <text x="{(buy_start+x1)/2}" y="34" font-family="IBM Plex Mono, monospace" font-size="11"
            fill="#9FB3AC" text-anchor="middle" letter-spacing="1">BUY ZONE</text>
      <line x1="{needle_x}" y1="32" x2="{needle_x}" y2="78" stroke="#EDEFE7" stroke-width="3" />
      <circle cx="{needle_x}" cy="{y}" r="7" fill="#EDEFE7" stroke="#0E1F1B" stroke-width="2" />
      <text x="{x0}" y="98" font-family="IBM Plex Mono, monospace" font-size="11" fill="#9FB3AC">0%</text>
      <text x="{sell_end}" y="98" font-family="IBM Plex Mono, monospace" font-size="11" fill="#9FB3AC" text-anchor="middle">48%</text>
      <text x="{buy_start}" y="98" font-family="IBM Plex Mono, monospace" font-size="11" fill="#9FB3AC" text-anchor="middle">60%</text>
      <text x="{x1}" y="98" font-family="IBM Plex Mono, monospace" font-size="11" fill="#9FB3AC" text-anchor="end">100%</text>
      <text x="{needle_x}" y="14" font-family="IBM Plex Mono, monospace" font-size="13" fill="#EDEFE7"
            font-weight="600" text-anchor="middle">P(up) = {prob*100:.1f}%</text>
    </svg>
    """
