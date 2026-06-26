"""Timing-alignment policy for feature construction.

This module documents the single, project-wide rule governing when an
exogenous series is observable at decision time. It holds no code: the only
helpers that implement the policy (``_safe``, ``_s1``, ``_mom_s1``,
``_diff_roll_s1``) are used solely inside
:func:`src.features.macro.add_macro_features`, so they remain nested closures
there rather than being hoisted into a shared module.

Policy
------
US-close exogenous series (VIX/MOVE/equity/commodity) are shifted by 1 day so a
feature is always observable at decision time. Same-day US-close values are safe
for US securities (SOFR) where feature and fixing are both US-close, but for
non-US fixings (EUR, NOR fix intraday before NY close) the same-day US close is
not yet available. Shifting US-close series by 1 day — feature[t] = series[t-1] —
makes every feature observable at any market.

Series NOT shifted (contemporaneous OK):
  - own-security swap rate: target and feature are the same daily close
  - country IBOR rates: same-country same-session fix
  - monthly macro: ffill+shift(1)

Series shifted by 1:
  - VIX, MOVE, V2X, VXN, RVX, OVX, GVZ
  - SPX, SX5E, MXWO, NDX
  - Oil, copper, OPEC, natgas
  - Breakeven inflation (TIPS), IG/HY credit spreads
  - Additional: CPURNSA, PCE CORE

Cross-market swap rates: shift(1) applied in
:func:`src.features.security.features_for_security`.
"""
