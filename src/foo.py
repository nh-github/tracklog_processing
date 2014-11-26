#! /usr/bin/env python
"""
...and, WHEEEEEE!
"""
import sys

#import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

from dateutil import parser

from osgeo import ogr

import gpxpy
import gpxpy.parser


def examine_gpx(fp):
    """
    ref: https://gis.stackexchange.com/questions/115868/\
            improving-this-script-gpx-reading-and-processing-\
            with-python-ogr/115987#115987
    """
    dataSource = ogr.Open(fp)
    layer = dataSource.GetLayerByName('track_points')
    first = layer.GetFeature(0)
    geom = first.GetGeometryRef()
    x, y, z = geom.GetPoint()
    print geom.GetGeometryName()
    print x, y


def foo_a(fp):
    """
    ref: https://gis.stackexchange.com/questions/115868/\
            improving-this-script-gpx-reading-and-processing-\
            with-python-ogr/115987#115987
    """
    dataSource = ogr.Open(fp)
    layer = dataSource.GetLayerByName('track_points')
    fig = plt.figure()
    ax = fig.gca()
    for i in range(layer.GetFeatureCount()):
        point = layer.GetFeature(i)
        geom_point = point.GetGeometryRef()
        x, y, z = geom_point.GetPoint()
        ax.scatter(x, y)
    plt.show()


def load_speeds(fp):
    print "load_speeds()"
    dataSource = ogr.Open(fp)
    layer = dataSource.GetLayerByName('track_points')

    print "loading points. .",
    point_list = []
    for i in range(1, layer.GetFeatureCount() - 1):
        p1 = layer.GetFeature(i - 1)
        p2 = layer.GetFeature(i)
        p3 = layer.GetFeature(i + 1)
        g1 = p1.GetGeometryRef()
        g2 = p2.GetGeometryRef()
        g3 = p3.GetGeometryRef()
        x1, y1, z1 = g1.GetPoint()
        x2, y2, z2 = g2.GetPoint()
        x3, y3, z3 = g3.GetPoint()
        d12 = distance_2d(y1, x1, y2, x2)
        d23 = distance_2d(y2, x2, y3, x3)

        t1 = parser.parse(p1.GetField("time"))
        t2 = parser.parse(p2.GetField("time"))
        t3 = parser.parse(p3.GetField("time"))
        td12 = t2 - t1
        td23 = t3 - t2
        td12_s = td12.total_seconds()
        td23_s = td23.total_seconds()
        s12 = d12 / td12_s  # UNITS: m/s
        s23 = d23 / td23_s
        s2avg = (s12 + s23) / 2

        point = {"lat": y2,
                 "lon": x2,
                 "time": t2,
                 "speed": s2avg,
                 }
        point_list.append(point)
        continue
        x = 3.6
        print ("{:3d}: {:0.2f} km/h"  # ", {:0.2f} m/s"
               ", {:0.2f} km/h"
               ", {:0.2f} km/h"
               #"; libcalc'd: {} m/s"
               "; d_12: {:0.2f} [meters]"
               "").format(i + 1,
                          s12 * x,
                          s23 * x,
                          s2avg * x,
                          #point_list[i].speed,
                          d12,
                          )
    print ".done"
    df = pd.DataFrame(point_list)
    extent = np.array(layer.GetExtent())
    lon_0 = np.mean(extent[0:2])
    lat_0 = np.mean(extent[2:])
    track_info = {"data": df,
                  "center": (lon_0, lat_0),
                  "lon_0": lon_0,
                  "lat_0": lat_0,
                  "bounds": extent
                  }
    return track_info


def foo_b(fp):
    """
    ref: https://stackoverflow.com/questions/15256638/ogr-distance-units
    """
    print "foo_b"
    dataSource = ogr.Open(fp)
    layer = dataSource.GetLayerByName('track_points')
    point_list = []
    for i in range(5):  # layer.GetFeatureCount() - 1):
        point = layer.GetFeature(i)
        p2 = layer.GetFeature(i + 1)
        g1 = point.GetGeometryRef()
        g2 = p2.GetGeometryRef()
        x1, y1, _ = g1.GetPoint()
        x2, y2, _ = g2.GetPoint()
        t1 = parser.parse(point.GetField("time"))
        t2 = parser.parse(p2.GetField("time"))
        td = t2 - t1
        td_s = td.total_seconds()
        #print g1.Distance(g2)
        dist = distance_2d(y1, x1, y2, x2)
        s23 = dist / td_s
        pt = {"i": i,
              "lon": x1,
              "lat": y1,
              "speed": (dist / td_s),
              }
        point_list.append(pt)

        if 0 < i and i < 5:
            print ("{:3d}: {:0.2f} km/h"  # ", {:0.2f} m/s"
                   #", {:0.2f} km/h"
                   #", {:0.2f} km/h"
                   #"; libcalc'd: {} m/s"
                   "").format(i + 1,
                              #s12 * x,
                              s23 * 3.6,
                              #s2avg * x,
                              #point_list[i].speed,
                              )
        #print dist, td,
        #print i, y1, x1, t1
        #print "    ", dist / td_s, "m/s"
        #print "    ", (dist / td_s) * 3.6, "km/h"
    return point_list


def foo_c(fp):
    """
    test gpxpy loading and processing

    refs:
        https://stackoverflow.com/questions/20308253/\
                gpx-parsing-calculate-speed-python
    """
    fd = open(fp, "r")
    gpx_parser = gpxpy.parser.GPXParser(fd)
    gpx_parser.parse()
    gpx = gpx_parser.get_gpx()
    #for track in gpx.tracks:
    gpx_part = gpx
    length_2d = gpx_part.length_2d()
    length_3d = gpx_part.length_3d()
    print "l2: ", length_2d
    print "l3: ", length_3d
    #(moving_time, stopped_time, moving_distance,
    # stopped_distance, max_speed) = gpx_part.get_moving_data()
    #print (moving_distance / 1000.) / (moving_time / 3600.)

    point_list = []
    for a, track in enumerate(gpx.tracks):
        for b, segment in enumerate(track.segments):
            for c, point in enumerate(segment.points):
                point_list.append(point)
    #i = 1
    #p1 = point_list[i + 0]
    #p2 = point_list[i + 1]
    #p3 = point_list[i + 2]
    x = (3600. / 1000)  # m/s -> km/h
    #print "index, n-1..n speed, n..n+1 speed, n-1..n+1 speed"
    for i in range(len(point_list)):
        try:
            p1 = point_list[i - 1]
            p2 = point_list[i + 0]
            p3 = point_list[i + 1]
        except IndexError:
            #point_list[i].speed = 0
            continue
        s12 = p2.speed_between(p1)
        s23 = p3.speed_between(p2)
        s2avg = np.mean(np.array([s12, s23]))
        s2mnl = p2.distance_3d(p1) / p2.time_difference(p1)  # m/s
        point_list[i].speed = s2avg
        if 0 < i and i < 5:
            print ("{:3d}: {:0.2f} km/h"  # ", {:0.2f} m/s"
                   ", {:0.2f} km/h"
                   ", {:0.2f} km/h"
                   #"; libcalc'd: {} m/s"
                   "; d_12: {:0.2f} [meters]"
                   "; foo: {:0.2f} [km/h]"
                   "").format(i + 1,
                              s12 * x,
                              s23 * x,
                              s2avg * x,
                              #point_list[i].speed,
                              p2.distance_3d(p1),
                              s2mnl * x,
                              )
        # if 0 == i % 5: print ""
    #for track in gpx.tracks:
    #    for segment in track.segments:
    #        for point_no, point in enumerate(segment.points):
    #            if point.speed is not None:
    #                print "{} Speed = {} m/s".format(point_no, point.speed)
    #            elif point_no > 0:
    #                print "{} Calc Speed = {:0.2f} m/s".format(point_no,
    #                                                           point.speed)
    #                print "Calculated speed =", point.speed_between(
    #                    segment.points[point_no - 1])
    return point_list


def split_track(track, geo_checks):
    df = track["data"]
    ms__kph = 3600. / 1000  # conversion for m/s to km/h
    sel = df["speed"] * ms__kph < 1.0
    df["paused"] = sel
    print df
    print df.ix[5]
    #x = np.arange(-3, 3, 0.1)
    #y = np.sin(x)
    #df = pd.DataFrame({"x": x, "y": y})
    #foo = df["y"]
    #bar = np.abs(foo) < .1
    #for i, pt in enumerate(point_list):
    #    kph = pt.speed * ms__kph
    #    if kph < 1.0:
    #        print i, kph
    plt.plot(df["lon"], df["lat"])
    plt.plot(df[sel]["lon"], df[sel]["lat"], 'bo')
    for check in geo_checks:
        print check
        plt.plot(check["lon"], check["lat"], 'rx', ms=15)
    plt.savefig("foo.pdf")


def distance_2d(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    modify to return meters!

    ref: https://gis.stackexchange.com/questions/61924/\
            python-gdal-degrees-to-meters-without-reprojecting
    originally relied on:
    #from math import cos, sin, asin, sqrt, radians
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 +\
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    #km = 6371 * c
    m = 6371 * c * 1000.
    return m


def check_plotting():
     #ortho, geos, and nsper

    m = Basemap(projection="merc",
                llcrnrlon=-157.864646,
                llcrnrlat=21.276013,
                urcrnrlon=-157.824974,
                urcrnrlat=21.318302,
                resolution="f")
    m.drawcoastlines()
    #m.drawmapboundary(fill_color='#00BBFF')
    #m.fillcontinents(color='coral', lake_color='aqua')
    #print m.xmax, m.xmin
    #x = np.arange(10000, 10)
    #y = x + 2000 * np.sin(x)

    lonpt, latpt = (-157.84, 21.28)
    xpt, ypt = m(lonpt, latpt)
    # convert back to lat/lon
    #lonpt, latpt = m(xpt,ypt,inverse=True)
    m.plot(xpt, ypt, 'bo')  # plot a blue dot there
    #m.plot(x, y)
    m.plot([1, 10, 100, 1000, 4000],
           [1000, 2000, 3000, 2000, 1000])
    #m.plot([0, 1, 10, 100, 1000], [0, 100, 200, 300, 500])
    #m.plot([-157.864646, -157.84, -157.824974],
    #       [21.276013, 21.28, 21.318302])
    plt.show()


def plot_map():
    # setup Lambert Conformal basemap.
    m = Basemap(width=12000000, height=9000000, projection='lcc',
                resolution='c', lat_1=45., lat_2=55, lat_0=50,
                lon_0=-107.)

    # draw coastlines.
    m.drawcoastlines()
    # draw a boundary around the map, fill the background.
    # this background will end up being the ocean color, since
    # the continents will be drawn on top.
    m.drawmapboundary(fill_color='#00BBFF')
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color='coral', lake_color='aqua')

    # draw parallels and meridians.
    # label parallels on right and top
    # meridians on bottom and left
    parallels = np.arange(0., 81, 10.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(10., 351., 20.)
    m.drawmeridians(meridians, labels=[True, False, False, True])
    plt.show()


def load_gpx(fp):
    dataSource = ogr.Open(fp)
    #layer = dataSource.GetLayerByName('tracks')
    layer = dataSource.GetLayerByName('track_points')
    print "tracks"
    numFeatures = layer.GetFeatureCount()
    extent = np.array(layer.GetExtent())
    print 'Feature count: ' + str(numFeatures)
    print 'Extent:', extent
    #print 'UL:', extent[0], extent[3]
    #print 'LR:', extent[1], extent[2]
    #print 'LL:', extent[0], extent[2]
    #print 'UR:', extent[1], extent[3]

    lon_0 = np.mean(extent[0:2])
    lat_0 = np.mean(extent[2:])

    points = []
    print "loading points. .",
    for i in range(numFeatures):
        point = layer.GetFeature(i)
        geom_point = point.GetGeometryRef()
        x, y, z = geom_point.GetPoint()
        t = parser.parse(point.GetField("time"))
        point = {"lon": x, "lat": y, "time": t}
        points.append(point)
        #help(geom_point.Distance)
        #sys.exit()
    print ".done"
    df = pd.DataFrame(points)
    track_info = {"data": df,
                  "center": (lon_0, lat_0),
                  "lon_0": lon_0,
                  "lat_0": lat_0,
                  "bounds": extent
                  }
    return track_info


def plot_track(trk):

    draw_marble = False
    df = trk["data"]
    extent = trk["bounds"]
    m = Basemap(projection="aeqd",  # "nsper", "ortho"
                lon_0=trk["lon_0"],
                lat_0=trk["lat_0"],
                llcrnrlon=extent[0],
                llcrnrlat=extent[2],
                urcrnrlon=extent[1],
                urcrnrlat=extent[3],
                resolution="f")
    if draw_marble:
        m.bluemarble()
    else:
        m.drawcoastlines()
        m.drawmapboundary(fill_color='#00BBFF')
        m.fillcontinents(color='coral', lake_color='aqua')

    lon_pts = df["lon"]
    lat_pts = df["lat"]
    x_pts, y_pts = m(lon_pts, lat_pts)
    #lonpt, latpt = m(xpt,ypt,inverse=True)  # convert back to lat/lon
    m.plot(x_pts, y_pts, 'bo')  # plot a blue dot there
    plt.show()


def check_gpx(fp):
    dataSource = ogr.Open(fp)
    for i in range(dataSource.GetLayerCount()):
        print dataSource.GetLayer(i).GetName(), " :",
        print dataSource.GetLayer(i).GetFeatureCount()


def load_gpx_track(fp):
    print "load_gpx_track()"
    dataSource = ogr.Open(fp)
    layer = dataSource.GetLayerByName('tracks')
    numFeatures = layer.GetFeatureCount()
    extent = np.array(layer.GetExtent())
    print 'Feature count: ' + str(numFeatures)
    print 'Extent:', extent
    #print 'UL:', extent[0], extent[3]
    #print 'LR:', extent[1], extent[2]
    #print 'LL:', extent[0], extent[2]
    #print 'UR:', extent[1], extent[3]

    lon_0 = np.mean(extent[0:2])
    lat_0 = np.mean(extent[2:])

    # geometry type
    first = layer.GetFeature(0)
    geom = first.GetGeometryRef()
    print geom.GetGeometryName()
    #print first.DumpReadable()  # string with lon/lat pairs
    #foo = geom.ExportToWkt()
    #print type(foo), (foo[:100])
    import json
    geomdb = json.loads(geom.ExportToJson())
    df = pd.DataFrame(geomdb['coordinates'][0], columns=['x', 'y'])


        #ExportToGML
        #ExportToJson
        #ExportToKML
        #ExportToWkb

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(df['x'], df['y'])
    plt.show()


    #tracks = []
    print "early return"
    return


def main():
    print "GPS tracklog work"
    check_points = [{"lat": 21.2873,
                     "lon": -157.8415,
                     "r": 50,  # 50 meters
                     "flag": True},  # within the range
                    {"lat": 21.2887,
                     "lon": -157.8355,
                     "r": 50,
                     "flag": True}]
    in_path = '/shared/media/gps/tracks-c_1QF102036/2014/09/20140923.gpx'
    in_path = '/shared/media/gps/tracks-c_1QF102036/2014/09/20140923a.gpx'
    #in_path = '/shared/media/gps/tracks-c_1QF102036/2014/10/20141002.gpx'
    check_gpx(in_path)
    load_gpx_track(in_path)
    #foo_a(in_path)
    #foo_b(in_path)
    #pl = foo_c(in_path)
    #split_track(pl)
    #check_plotting()
    #trk = load_gpx(in_path)
    #trk = load_speeds(in_path)
    #plot_track(trk)
    #split_track(trk, geo_checks=check_points)

if "__main__" == __name__:
    main()
    sys.exit()
