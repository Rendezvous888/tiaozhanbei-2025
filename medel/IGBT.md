# IGBT Key Parameters (FF1500R17IP5R)

## Electrical Characteristics:
- **Collector-Emitter Voltage (VCES)**: 1700V
- **Nominal Collector Current (IC nom)**: 1500A
- **Repetitive Collector Peak Current (ICRM)**: 3000A
- **Gate-Emitter Peak Voltage (VGES)**: ±20V
- **Collector-Emitter Saturation Voltage (VCE sat)**:
  - 1.75V @ IC = 1500A (Tvj = 25°C)
  - 2.10V @ IC = 1500A (Tvj = 125°C)
  - 2.30V @ IC = 1500A (Tvj = 175°C)
- **Gate Threshold Voltage (VGEth)**: 
  - Min: 5.35V
  - Typ: 5.80V
  - Max: 6.25V
  - @ IC = 54.0 mA, VCE = VGE, Tvj = 25°C
- **Gate Charge (QG)**: 7.50 µC @ VGE = ±15V, VCE = 900V
- **Input Capacitance (Cies)**: 88.0 nF @ VCE = 25V, f = 1000 kHz
- **Reverse Transfer Capacitance (Cres)**: 2.70 nF @ VCE = 25V, f = 1000 kHz
- **Collector-Emitter Cut-Off Current (ICES)**: 10 mA @ VCE = 1700V, VGE = 0V, Tvj = 125°C
- **Gate-Emitter Leakage Current (IGES)**: 400 nA @ VCE = 0V, VGE = 20V, Tvj = 25°C

## Switching Characteristics:
- **Turn-On Delay Time (td on)**: 0.30 µs @ IC = 1500A, VCE = 900V, VGE = ±15V
- **Rise Time (tr)**: 0.15 µs @ IC = 1500A, VCE = 900V, VGE = ±15V
- **Turn-Off Delay Time (td off)**: 0.66 µs @ IC = 1500A, VCE = 900V, VGE = ±15V
- **Fall Time (tf)**: 0.11 µs @ IC = 1500A, VCE = 900V, VGE = ±15V

## Energy Losses:
- **Turn-On Energy Loss (Eon)**:
  - Min: 335 mJ @ IC = 1500A, VCE = 900V
  - Max: 595 mJ @ IC = 1500A, VCE = 900V
- **Turn-Off Energy Loss (Eoff)**:
  - Min: 330 mJ @ IC = 1500A, VCE = 900V
  - Max: 545 mJ @ IC = 1500A, VCE = 900V

## Thermal Characteristics:
- **Thermal Resistance (RthJC)**: 19.0 K/kW (junction to case)
- **Operating Junction Temperature (Tvj op)**: -40°C to 175°C
- **Thermal Resistance (RthCH)**: 15.5 K/kW (case to heatsink)
- **Thermal Resistance (RthJC Diode)**: 35.0 K/kW (junction to case per diode)
