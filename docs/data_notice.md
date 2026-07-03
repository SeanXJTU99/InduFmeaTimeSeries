# Data Anonymization Notice

## Fictitious Data

All PLC tag names, valve identifiers, distillation column designators,
serial device addresses, and FMEA entries in this repository are **entirely
fictitious**.  They do not correspond to any real industrial facility,
equipment, or process.

## Tag Naming Convention

All tags follow the pattern `<type>-<3-digit-number>`:

| Prefix | Type | Example |
|--------|------|---------|
| TE | Temperature Element | TE-301 |
| PT | Pressure Transmitter | PT-301 |
| FT | Flow Transmitter | FT-301 |
| FV | Flow Control Valve | FV-301 |
| LT | Level Transmitter | LT-301 |

## Column Designators

- T-301, T-302, T-303 — fictitious cryogenic distillation columns
- E-301 — fictitious heat exchanger
- H-301 — fictitious reboiler
- C-301 — fictitious condenser
- F-101 — fictitious feed system

## Mock Data

Files in `data/mock/` contain synthetic data generated for testing and
demonstration.  They do not represent any real industrial process.

## FMEA Entries

All FMEA (Failure Mode and Effects Analysis) entries are invented for
demonstration.  Severity (S), Occurrence (O), Detection (D), and Risk
Priority Number (RPN) values are illustrative only.

## Compliance

This repository is intended as a portfolio demonstration of AI/ML
engineering capability.  It does not contain, and has never contained,
any proprietary industrial data, trade secrets, or real equipment
specifications.
