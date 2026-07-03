# Mock Data

Fictitious sample data for testing and demonstration.  None of these values
represent any real industrial process, equipment, or facility.

## Files

| File | Description |
|------|-------------|
| `fmea_sample.json` | 8 fictitious FMEA entries across 4 equipment types |
| `plc_stream_sample.csv` | 100 rows of simulated PLC time-series (1-second resolution) |
| `serial_stream_sample.bin` | 256 bytes of simulated RS485 binary telemetry |

## Usage

```python
import json
with open("data/mock/fmea_sample.json") as f:
    fmea = json.load(f)
```
