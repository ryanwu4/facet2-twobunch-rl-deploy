"""Stand-in for the AIDA-PVA multiknob service, for the offline tests: serves the RPC actuate.turn_knob() calls
(pvs.MULTIKNOB_SERVICE) over PVAccess. A call must name pvs.MULTIKNOB, as the real service needs the knob's name. It
turns the soft IOC's LI12-14 subbooster setpoints by the delta over Channel Access, as the real knob moves them 1:1, and
replies with the delta. Prints "ready", then serves until killed.

    python tests/mkb_server.py

The test sets the PVAccess and Channel Access addresses and ports through the environment, all on 127.0.0.1.
"""
import time

import epics
from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV

from twobunch_deploy import pvs

service = SharedPV(nt=NTScalar("d"), initial=0.0)


@service.rpc
def turn(pv, op):
    query = op.value().query
    if query.MKB != pvs.MULTIKNOB:
        op.done(error=f"unknown multiknob {query.MKB}")
        return
    delta = float(query.VALUE)
    for name in pvs.MULTIKNOB_MOVES:
        epics.caput(name, epics.caget(name) + delta, wait=True)
    op.done(NTScalar("d").wrap(delta))


with Server(providers=[{pvs.MULTIKNOB_SERVICE: service}]):
    print("ready", flush=True)
    while True:
        time.sleep(1)
