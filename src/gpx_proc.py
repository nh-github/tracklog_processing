#! /usr/bin/env python
"""
Copy interesting subset of tracklog from gpx file

Specify input file path (autogenerate output path)
TODO: specify check points
"""

__version__ = 0.7

import argparse
import datetime
import logging
import os
import sys

#import matplotlib as mpl
import numpy as np
import pandas as pd
import gpxpy
import gpxpy.parser


class gpx_proc(object):
    """
    extract an ``interesting'' segment from a tracklog file

    call gpx_proc().proc_file(<input>, <output>, <geographic points>)
    * input and output are paths to gpx files (output is created)
    * geographic points are latitude/longitude to guide selection of
        the interesting segment
    """

    def proc_file(self, inpath, outpath, checkpoints):
        """
        merge/trim to single track between pauses in motion at checkpoints

        Process:
        * open files
        * load data
        * check parameters (speed, location flags)
        * create a subset
        * clean track
        * write data
        * close files
        """
        logging.info("IN")
        ifd = open(inpath, "r")
        track_data = self.load_file(ifd)
        slow_indices = self.check_speed(track_data)  # (slow is <1 km/h)
        check_indices = self.check_loc_points(track_data.tracks[0],
                                              checkpoints)
        endpoints = self.select_subset(track_data.tracks[0], slow_indices,
                                       check_indices)
        if endpoints:
            extracted_track = self.clip_track(track_data.tracks[0], endpoints)
            #TODO: re-runnable without creating many files
            ofd = open(self.make_outpath(inpath, outpath), "w")
            self.save_file(ofd, extracted_track)
        else:
            logging.error("No usable subset found")
        logging.info("OUT")
        return

    def select_subset(self, gpx_track, slow_indices, check_indices):
        points_list = gpx_track.segments[0].points
        empty_col = np.zeros(len(points_list))
        df = pd.DataFrame({"points": points_list,
                           "slow_flags": empty_col,
                           "check_flags": empty_col,
                           })
        df["slow_flags"][slow_indices] = 1
        df["check_flags"][check_indices] = 1
        end_pairs = []
        end_pair = []
        for index, point in enumerate(points_list):
            if 0 == df["check_flags"][index]:
                continue  # ignore possible endpoints away from checkpoints
            if(0 == df["slow_flags"][index + 1] and
               1 == df["slow_flags"][index]):
                logging.warn("start: {}".format(index))
                end_pair = [index]
            if(1 == df["slow_flags"][index + 1] and
               0 == df["slow_flags"][index] and
               1 == len(end_pair)):
                logging.warn(" stop: {}".format(index))
                end_pair.append(index)
                end_pairs.append(end_pair)
                end_pair = []

        #just take the longest span
        logging.debug("find longest span..")
        longest_pair = []
        longest_time = datetime.timedelta(0)
        for pair in end_pairs:
            p0 = points_list[pair[0]]
            p1 = points_list[pair[1]]
            elapsed = p1.time - p0.time
            logging.debug("{}..{}, {}; {}".format(p0.time, p1.time,
                                                  elapsed, pair))
            if longest_time < elapsed:
                longest_time = elapsed
                longest_pair = pair

        logging.debug("longest span: {}, over {}s".format(longest_pair,
                                                          longest_time))

        if longest_time > datetime.timedelta(0, 60):
            logging.info("{}".format(longest_pair))
        else:
            longest_pair = None
        logging.info("OUT")
        return longest_pair

    def save_file(self, ofd, gpx_obj):
        logging.info("IN")
        ofd.write(gpx_obj.to_xml())
        logging.info("OUT")
        return

    def load_file(self, ifd):
        """
        load as a single track
        """
        logging.info("IN")
        gpx_parser = gpxpy.parser.GPXParser(ifd)
        gpx_parser.parse()
        parsed_gpx = gpx_parser.get_gpx()

        # merge all tracks and track segments into a single track segment
        new_gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        new_gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        for track in parsed_gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    gpx_segment.points.append(point)

        logging.info("OUT")
        return new_gpx

    def make_outpath(self, inpath, outpath):
        """
        Generate an output path based on input path if needed/requested
        """
        logging.info("IN")
        if outpath is not None:
            logging.info("OUT")
            candidate = outpath
        else:
            d, f = os.path.split(inpath)
            b, e = os.path.splitext(f)
            candidate = inpath
            i = 0
            while os.path.exists(candidate):
                mod = chr(i + ord("a"))
                i += 1
                candidate = os.path.join(d, b + "-" + mod + e)

        logging.info("OUT")
        return candidate

    def clip_track(self, gpx_track, endpoints):
        """
        combine gpx_track and extra info to extract interesting section
        """
        logging.info("IN")
        gpx_subset = gpx_track.clone()
        gpx_subset.split(0, endpoints[1])
        gpx_subset.split(0, endpoints[0])

        # Create a new record
        gpx_data = gpxpy.gpx.GPX()

        # Create a track
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_data.tracks.append(gpx_track)

        # Add the segment of interest to the track
        #gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_subset.segments[1])

        #gpx_track.simplify()  # simple copy for now

        logging.info("OUT")
        return gpx_data

    def check_loc_points(self, gpx_track, loc_points):
        """
        find points within the specified range of location(s)

        return a list of indices for the points
        """
        logging.info("IN")
        points_list = gpx_track.segments[0].points
        close_indices = []
        for loc_pt in loc_points[:4]:
            print ("LOC: {lon:0.5f} {lat:0.5f} "
                   "[{label}] r: {r:} m").format(**loc_pt)
            cp = gpxpy.gpx.GPXTrackPoint(loc_pt["lat"],
                                         loc_pt["lon"],
                                         name=loc_pt["label"])
            for i, point in enumerate(points_list):
                d = cp.distance_2d(point)
                if d < loc_pt["r"]:
                    close_indices.append(i)
                    continue
                    print ("{} "
                           "{:0.5f} {:0.5f} "
                           "closeness: {:0.2f} "
                           "t: {}"
                           "").format(i,
                                      point.longitude,
                                      point.latitude,
                                      cp.distance_2d(point),
                                      point.time,)
        #print points_list[1]
        #p1 = points_list[509]
        #p2 = points_list[511]
        #print p1.distance_3d(p2)
        #print p1.speed_between(p2)
        #print p1.speed_between(p2) * 3.6
        #print help(gpx_track.get_points_data)
        #print gpx_track.get_points_no()
        #print help(gpx_track.get_point)
        #print help(gpx_track.get_nearest_location)
        logging.info("OUT")
        return close_indices

    def check_speed(self, gpx_data, slow_speed=1.0):
        """
        Calculate speed over track and index slow points (list)

        NOTE: only checks first track segment (of first track)

        refs:
            https://stackoverflow.com/questions/20308253/\
                    gpx-parsing-calculate-speed-python
        """
        logging.info("IN")
        logging.info("Overall length: {:0,.2f} meters"
                     "".format(gpx_data.length_3d()))

        point_list = gpx_data.tracks[0].segments[0].points
        speed_units = (3600. / 1000)  # m/s -> km/h
        for i in range(len(point_list)):
            try:
                p1 = point_list[i - 1]
                p2 = point_list[i + 0]
                p3 = point_list[i + 1]
            except IndexError:
                continue
            s12 = p2.speed_between(p1)
            s23 = p3.speed_between(p2)
            s2avg = np.mean(np.array([s12, s23]))
            s2mnl = p2.distance_3d(p1) / p2.time_difference(p1)  # m/s
            point_list[i].speed = s2avg
            if False and 0 < i and i < 5:
                print ("{:3d}: {:0.2f} km/h"  # ", {:0.2f} m/s"
                       ", {:0.2f} km/h, {:0.2f} km/h"
                       "; d_12: {:0.2f} [meters]; foo: {:0.2f} [km/h]"
                       "").format(i + 1,
                                  s12 * speed_units, s23 * speed_units,
                                  s2avg * speed_units, p2.distance_3d(p1),
                                  s2mnl * speed_units,)
        slow_indices = []
        for i, point in enumerate(point_list):
            point_kph = point.speed * speed_units
            if point_kph < slow_speed:
                slow_indices.append(i)
                continue  # TODO: skip extra printing
                print ("point #{} {:0.2f} km/h, {:0.4f} / {:0.4f}, {}"
                       "").format(i, point_kph,
                                  point.longitude,
                                  point.latitude,
                                  point.time)
        logging.info("OUT")
        return slow_indices


def parse_command_line(command_line):
    parser = argparse.ArgumentParser(description="GPS tracklog work")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    parser.add_argument("-q", "--quiet", action="count", default=0,
                        dest="unverbosity", help="quiet output")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    #-h/--help is auto-added
    parser.add_argument("-i", "--in", action="store", default=None,
                        required=True, help="input path (required)")
    # TODO: default output path based on to input path
    parser.add_argument("-o", "--out", action="store")
    # TODO: allow check point specification on command line.
    #   allow multiple points added, one per argument
    parser.add_argument("-p", "--check-point", action="store")
    ret = vars(parser.parse_args())
    return ret


def setup_logging():
    log_fmt = ("%(levelname)s - %(module)s - %(name)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    log_fmt = ('%(levelname)s - '
               '%(funcName)s @%(lineno)d: %(message)s')
    #addl keys: asctime, module, name
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.DEBUG)
    return


def main():
    cmd_args = parse_command_line(sys.argv)
    setup_logging()
    check_points = [{"label": "Mc Cully launch site (shore)",
                     "lat": 21.2884,
                     "lon": -157.8323,
                     "r": 40,  # meters
                     "flag": True},  # within the range
                    {"label": "Library launch site (shore)",
                     "lat": 21.2760,
                     "lon": -157.8183,
                     "r": 50,
                     "flag": True}]

    in_path = cmd_args["in"]
    out_path = cmd_args["out"]
    gpx_proc().proc_file(in_path, out_path, check_points)

if "__main__" == __name__:
    main()
    sys.exit()
