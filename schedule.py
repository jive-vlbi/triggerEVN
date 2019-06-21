#!/usr/bin/env python

import keyin_template
from log2vex.vex import vex
from log2vex import vex_util, merge, clock_model

from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astroquery.nrao import Nrao

import numpy

import logging
from datetime import datetime, timedelta
import itertools
import tempfile
import os.path
import subprocess
import shlex
import os
import copy
import math

def make_schedule(experiment, stations, 
                  center_coordinate, observing_frequency, 
                  frequency_low=None, frequency_up=None, 
                  start_time=timedelta(minutes=5), 
                  duration=timedelta(minutes=60), 
                  time_on_calibrator=timedelta(minutes=2), 
                  time_on_sources=timedelta(minutes=5),
                  gap_time=timedelta(seconds=30),
                  max_sources=10, min_rms=None, 
                  calibrator_radius=1 * units.deg,
                  targets_radius=1 * units.deg,
                  phase_center_name="PHASE_CENTER"):
    """
    Returns two VEX file names as tuple: (only phase center, with all sources)
    for station, correlator usage respectively.
    @center_coordinate is an astropy.coordinates.SkyCoord 
    @stations is an iterable of station names as recognized by SCHED
    @start_time is a datetime (absolute time) or timedelta (added to now)
    the frequencies and @min_rms are astropy.quantity.Quantity instances
    @frequency_low and @frequency_up are passed on to the query of the NRAO 
     database, if None they default to observing_frequency * 0.5 and
                                       observing_frequency * 1.5 + bandwidth.
     Only sources whose observing frequency are completly within this range
     will be returned.
    """
    bandwidth = 128 * units.MHz # matches the (hardcoded) schedule
    if frequency_low is None:
        frequency_low = observing_frequency * 0.5
    if frequency_up is None:
        frequency_up = observing_frequency * 1.5 + bandwidth
    if isinstance(start_time , timedelta):
        start_time = datetime.utcnow()
    start_time = start_time.replace(microsecond=0)
    
    sources_table = Nrao.query_region(center_coordinate, targets_radius,
                                      freq_low=frequency_low, 
                                      freq_up=frequency_up)
    sources_table.sort("Image RMS")
    sources = {} # {name: (ra, dec)} with ra and dec
    for row in sources_table:
        if (min_rms is not None) and (row["Image RMS"] * units.mJy < min_rms):
            break
        name = row["Source"].decode()
        name = name.replace(" ", "_")
        if name not in sources:
            sources[name] = (Angle(row["RA"].decode()), 
                             Angle(row["DEC"].decode()))
            if len(sources) >= max_sources:
                break

    if len(sources) == 0:
        logging.warning("No known matching sources.")

    calibrator, calibrator_coord = get_calibrator(
        center_coordinate, calibrator_radius)

    # SCHED doesn't support multiple sources per scan, so only schedule 
    # phase center and calibrator
    def to_string(angle):
        return angle.to_string(units.hour, sep=":")
    def to_string_deg(angle):
        return angle.to_string(units.deg, sep=":")
    sources_text = \
                   keyin_template.source.format(
                       name=calibrator, 
                       ra=to_string(calibrator_coord.ra), 
                       dec=to_string_deg(calibrator_coord.dec)) + \
                   keyin_template.source.format(
                       name=phase_center_name, 
                       ra=to_string(center_coordinate.ra), 
                       dec=to_string_deg(center_coordinate.dec))
    
    # fill the available time with calibrator and phase center scans
    scans_text = ""
    scan_time = start_time
    end_time = start_time + duration
    for (source, scan_duration) in itertools.cycle(
            ((calibrator, time_on_calibrator), 
             (phase_center_name, time_on_sources))):
        scan_end = scan_time + scan_duration
        if scan_end >= end_time:
            scan_duration = end_time - scan_time
        scans_text += keyin_template.scan.format(
            source=source,
            seconds=scan_duration.total_seconds(),
            gap=gap_time.total_seconds())
        scan_time = scan_end + gap_time
        if scan_time >= end_time:
            break

    keyin_input = keyin_template.sched.format(
        exp=experiment,
        sources=sources_text,
        freq_MHz=(observing_frequency/units.MHz).value, 
        year=start_time.year,
        month=start_time.month,
        day=start_time.day,
        time=start_time.time(),
        stations=", ".join(stations),
        scans=scans_text)

    vex_file_name = make_vex(keyin_input, experiment)

    vix_file_name = make_vix(vex_file_name, experiment, sources, 
                             phase_center_name)

    return (vex_file_name, vix_file_name)

sched_location = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              "sched")

def get_calibrator(coordinate, radius):
    # import keyin reader from SCHED source
    import sys
    old_path = copy.copy(sys.path)
    try:
        sys.path.append(os.path.join(sched_location, "src", "python"))
        import key
    finally:
        sys.path = old_path
    sources = os.path.join(sched_location, "catalogs", "sources.rfc")
    with open(sources, 'r') as f:
        calibrators = key.read_keyfile(f)
        pass
    ra = []
    dec = []
    flux = []
    name = []
    for calibrator in calibrators:
        if not 'FLUX' in calibrator:
            continue
        name.append(calibrator['SOURCE'][1])
        ra.append(calibrator['RA'] * math.pi / (12 * 60 * 60))
        dec.append(calibrator['DEC'] * math.pi / (180 * 60 * 60))
        flux.append(calibrator['FLUX'][1])
        continue
    catalog = SkyCoord(ra=ra, dec=dec, unit="rad")
    # Position of target
    d2d = coordinate.separation(catalog)
    catalogmask = d2d < radius
    idxcatalog = numpy.where(catalogmask)[0]
    best_flux = 0
    best_idx = None
    for idx in idxcatalog:
        if flux[idx] > best_flux:
            best_flux = flux[idx]
            best_idx = idx
            pass
    return (name[best_idx], catalog[best_idx])

def make_vex(keyin_input, experiment):
    # run the keyin input through SCHED in a temporary directory
    tmp_dir_name = tempfile.mkdtemp()

    keyin_file_name = os.path.join(tmp_dir_name, "key.in")
    with open(keyin_file_name, "w") as keyin_file:
        keyin_file.write(keyin_input)
    command = "{sched}/bin/sched.py -k {sched_file}".format(
        sched=sched_location, sched_file=keyin_file_name)

    sched_env = copy.copy(os.environ)
    sched_env["SCHED"] = sched_location
    subprocess.check_call(shlex.split(command), 
                          cwd=tmp_dir_name, 
                          env=sched_env)
    vex_file_name = os.path.join(tmp_dir_name, experiment.lower() + ".vex")
    return vex_file_name

def make_vix(vex_file_name, experiment, sources, phase_center_name):
    # create multi phase center, correlator VEX file
    vex_struct = vex.Vex(vex_file_name)
    vex_util.y2yy(vex_struct)
    merge.generate_vix(
        experiment, vex_struct, evlbi=True, from_skd=True, logfiles=None,
        clock_models=clock_model.get_clock_models(vex_struct))

    # add sources to the source block
    source_block = vex_struct["SOURCE"]
    source_template = """
def {name};
     source_name = {name};
     ra = {ra}; dec = {dec}; ref_coord_frame = J2000;
enddef;"""
    for name, (ra, dec) in sources.items():
        source_block.insert_last(vex.parse_def(source_template.format(
            name=name, 
            ra=ra.to_string(units.hour, sep="hms", pad=True), 
            dec=dec.to_string(units.deg, sep="d'\"", pad=True))))

    # add the sources to the phase center scans
    for source in sources.keys():
        new_line = " source={};\n".format(source)
        sched_block = vex_struct["SCHED"]
        for scan in sched_block.values():
            value = scan["source"]
            if value == phase_center_name:
                scan.insert_after("source", value, vex.parse_lines(new_line))

    vix_file_name = vex_file_name[:-3] + "vix"
    with open(vix_file_name, "w") as f:
        f.write(vex.vex_element_to_str(vex_struct))
    return vix_file_name


#SkyCoord(76.5845, 56.7246, frame='fk5', unit='deg'),
if __name__ == "__main__":
    print(make_schedule("FR059", 
                        ["ONSALA85", "WSTRBORK", "HART"], 
                        SkyCoord("00h19m37.8544967s", "20d21m45.644559s"),
#                        SkyCoord("03h14m46.3s", "20d14m20.4s"),
#                        SkyCoord("00h21m08.143s", "20d58m17.41s"),
                        4926.49 * units.MHz,
                        calibrator_radius = 5 * units.deg,
                        targets_radius=1 * units.deg,
                        start_time=datetime(2019,4,29,7,00,0),
                        duration=timedelta(minutes=300),
                        time_on_calibrator=timedelta(minutes=10),
                        time_on_sources=timedelta(minutes=5),
                        max_sources=0))
