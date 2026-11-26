"""Read the supported horizontal storage vessel dimensions."""

# The R1 installed on the second floor of D02 defines the production model.
HORIZONTAL_VESSEL_TIER = 'small'


def is_horizontal(vessel):
    return vessel.get('form', vessel.get('orientation')) == 'horizontal'


def normalize_vessel(vessel):
    """Apply current equipment selection to a working copy of the source recipe."""
    if is_horizontal(vessel):
        vessel.update(size_tier=HORIZONTAL_VESSEL_TIER, reactor=False, jacket=False)
        vessel.pop('jacket_medium', None)
        vessel.pop('reflux_condenser', None)


def storage_service_checks(graph, actual=None):
    """Check ownership in the process graph and, if supplied, native model parts."""
    rows = []
    units = graph['units']
    for uid, vessel in units.items():
        if vessel.get('kind') != 'vessel' or not is_horizontal(vessel):
            continue
        jackets = {key for key, unit in units.items()
                   if unit['kind'] == 'jacket' and uid in (unit.get('owner'), unit.get('integral_to'))}
        condensers = {key for key, unit in units.items()
                      if unit['kind'] == 'condenser' and unit.get('service') == 'reflux'
                      and uid in (unit.get('owner'), unit.get('service_owner'))}
        condensers.update(c['condenser'] for c in graph.get('reflux_circuits', []) if c['owner'] == uid)

        # The standard root geom / nozzle names also catch parts left in the
        # XML after removal of their logical equipment from the graph.
        physical = []
        if actual is not None:
            ids = {f'CND_{uid}', *condensers}
            for jacket in {f'J_{uid}', *jackets}:
                ids.update(f'{jacket}_{port}_equipment_nozzle' for port in ('utility_in', 'utility_out'))
            physical = sorted(name for name in ids if name in actual)
        fail = vessel.get('reactor') or vessel.get('jacket') or jackets or condensers or physical
        rows.append({'equipment': uid, 'status': 'fail' if fail else 'pass',
                     'reactor': bool(vessel.get('reactor')), 'jacket': bool(vessel.get('jacket')),
                     'jackets': sorted(jackets), 'reflux_condensers': sorted(condensers),
                     'forbidden_xml_components': physical if actual is not None else None})
    return rows
