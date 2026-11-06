# DN25 handle ball valve

A generic simulation model based on selected dimensions of the **Bray Flow-Tek 7000 series / 7100 threaded full-port DN25 valve**. It uses a raised flat metal handle with a thin blue grip cover.

## Main dimensions

| Parameter | Value | Source |
|---|---|---|
| Body length | 92 mm | Bray catalogue, A |
| Stem axis to handle end | 200 mm | Bray catalogue, D |
| Pipe axis to highest point | 87 mm | Bray catalogue, C |
| Pipe axis to mounting top | 52 mm | Bray catalogue, C1 |
| Pipe outside diameter / section length | 33.7 mm / 700 mm | Task settings |
| Handle width / covered thickness / metal plate thickness | 20 mm / 6 mm / 3 mm | Task settings |

## Resistance parameters

| Stage | Reference torque |
|---|---:|
| Breakaway | 14 N·m |
| Running | 10.5 N·m |
| Near the endpoint | 12.6 N·m |

Based on Bray TB1005's DN25 full-port, RPTFE-seat, 0 bar reference and typical 100/75/90% torque ratios.

The station XML files hold a fixed friction setting of **14 N·m**. Applying the full resistance profile requires runtime code that is not included in these model files; the parameters are recorded in JSON.

## Files and formats

| Format | Contents |
|---|---|
| MuJoCo MJCF (`.xml`) | Valve, pipe, and supports in the six [A1–C2 station models](../../scenes/handle-ball-valve-stations/) |
| JSON (`.json`) | [Valve dimensions and resistance parameters](parameters.json); companion station files describe placement and axes |

Valve geometry is included in the station XML files. OpenUSD and CAD exports are not yet available.

## References

- [Bray 7000/8000 brochure](https://www.bray.com/docs/default-source/brochures/product-brochures/en_f-2301_s7000_8000_web.pdf): page 10, DN25 dimensions; page 11, 7100 threaded-end model.
- [Bray TB1005](https://www.bray.com/docs/default-source/technical-bulletin/en_tb-1005_act-selectguide-ball-valves.pdf): page 3, torque reference; page 11, typical torque ratios.

## License

Our model parameters and station XML files are **Apache-2.0**; this guide is **CC BY 4.0**. The linked Bray documents retain their original rights. See [license scope](../../../LICENSING.md) and [third-party notices](../../../THIRD_PARTY_NOTICES.md).
